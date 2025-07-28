/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
    obtain(chunk, geom.is_slice_rendering_ptr, 1, 128);
    obtain(chunk, geom.slice_params_ptr, 5, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug,
    const bool is_slice_rendering,
    const float z_position,
    const float x_min,
    const float x_max,
    const float y_min,
    const float y_max)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    CHECK_CUDA(cudaMemcpy(geomState.is_slice_rendering_ptr, &is_slice_rendering, sizeof(bool), cudaMemcpyHostToDevice), debug);
    float slice_params[5] = {z_position, x_min, x_max, y_min, y_max};
    CHECK_CUDA(cudaMemcpy(geomState.slice_params_ptr, slice_params, 5 * sizeof(float), cudaMemcpyHostToDevice), debug);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

    if (!is_slice_rendering)
    {
        CHECK_CUDA(FORWARD::preprocess(
            P, D, M,
            means3D, (glm::vec3*)scales, scale_modifier, (glm::vec4*)rotations,
            opacities, shs, geomState.clamped, cov3D_precomp, colors_precomp,
            viewmatrix, projmatrix, (glm::vec3*)cam_pos,
            width, height, focal_x, focal_y, tan_fovx, tan_fovy,
            radii, geomState.means2D, geomState.depths, geomState.cov3D,
            geomState.rgb, geomState.conic_opacity, tile_grid,
            geomState.tiles_touched, prefiltered, antialiasing
        ), debug);
    }
    else
    {
        CHECK_CUDA(FORWARD::preprocess_slice(
            P,
            means3D, (glm::vec3*)scales, scale_modifier, (glm::vec4*)rotations,
            opacities, cov3D_precomp, colors_precomp,
            width, height, z_position, x_min, x_max, y_min, y_max,
            radii, geomState.means2D, geomState.depths, geomState.cov3D,
            geomState.rgb, geomState.conic_opacity,
            tile_grid, geomState.tiles_touched, antialiasing
        ), debug);
    }

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P, geomState.means2D, geomState.depths, geomState.point_offsets,
		binningState.point_list_keys_unsorted, binningState.point_list_unsorted,
		radii, tile_grid);
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space, binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered, binningState.point_list_keys, imgState.ranges);
	CHECK_CUDA(, debug)

	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block, imgState.ranges, binningState.point_list,
		width, height, geomState.means2D, feature_ptr,
		geomState.conic_opacity, imgState.accum_alpha, imgState.n_contrib,
		background, out_color, geomState.depths, depth), debug)

	return num_rendered;
}

void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* image_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(image_buffer, width * height);

    bool is_slice_rendering;
    CHECK_CUDA(cudaMemcpy(&is_slice_rendering, geomState.is_slice_rendering_ptr, sizeof(bool), cudaMemcpyDeviceToHost), debug);

	const int* final_radii = (radii == nullptr) ? geomState.internal_radii : radii;

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid, block,
		imgState.ranges, binningState.point_list,
		width, height, background,
		geomState.means2D, geomState.conic_opacity,
		color_ptr, geomState.depths,
		imgState.accum_alpha, imgState.n_contrib,
		dL_dpix, dL_invdepths,
		(float3*)dL_dmean2D, (float4*)dL_dconic,
		dL_dopacity, dL_dcolor, dL_dinvdepth), debug);

	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    if (!is_slice_rendering)
    {
        CHECK_CUDA(BACKWARD::preprocess(P, D, M,
            (float3*)means3D, final_radii, shs, geomState.clamped, opacities,
            (glm::vec3*)scales, (glm::vec4*)rotations, scale_modifier,
            cov3D_ptr, viewmatrix, projmatrix,
            focal_x, focal_y, tan_fovx, tan_fovy, (glm::vec3*)campos,
            (float3*)dL_dmean2D, dL_dconic, dL_dinvdepth, dL_dopacity,
            (glm::vec3*)dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh,
            (glm::vec3*)dL_dscale, (glm::vec4*)dL_drot,
            antialiasing), debug);
    }
    else
    {
        float slice_params[5];
        CHECK_CUDA(cudaMemcpy(slice_params, geomState.slice_params_ptr, 5 * sizeof(float), cudaMemcpyDeviceToHost), debug);
        float z_position = slice_params[0];
        float x_min = slice_params[1];
        float x_max = slice_params[2];
        float y_min = slice_params[3];
        float y_max = slice_params[4];

        CHECK_CUDA(BACKWARD::preprocess_slice(
            P,
            (float3*)means3D, final_radii, opacities,
            (glm::vec3*)scales, (glm::vec4*)rotations,
            scale_modifier, cov3D_ptr,
            z_position, x_min, x_max, y_min, y_max,
            (float)width, (float)height,
            (const float3*)dL_dmean2D,
            dL_dconic,
            dL_dopacity,
            (glm::vec3*)dL_dmean3D,
            dL_dcov3D,
            (glm::vec3*)dL_dscale,
            (glm::vec4*)dL_drot,
            antialiasing
        ), debug);
    }
}