# Gaussian Slicing for Volumetric Data

This project is a significant modification of the original **[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)** by Inria. The core rendering pipeline has been fundamentally adapted to reconstruct and render 3D volumetric data from a series of orthogonal 2D slices (e.g., from confocal microscopy or MRI scans), instead of standard perspective images.

## Key Modifications

The primary goal of this fork is to enable 3D Gaussian Splatting to work with scientific imaging data that comes in the form of co-aligned stacks of 2D images.

- **Data Loading Pipeline (`/scene`)**: The original data loaders, which rely on COLMAP for camera pose estimation, have been replaced.
  - A new `SlicePlane` object (`scene/camera.py`) was created to represent a 2D slice with a specific Z-position.
  - A custom loader (`scene/slice_loader.py`) now reads a directory of 2D grayscale images and a sparse point cloud to initialize the scene.

- **Gaussian Model (`/scene/gaussian_model.py`)**: The Gaussian representation has been simplified for this specific application.
  - View-dependent color representation using Spherical Harmonics (SH) has been completely removed.
  - Gaussians now have a single-channel feature representing intrinsic intensity, greatly reducing model complexity.

- **CUDA Rendering Pipeline (`/submodules/diff-gaussian-rasterization`)**: This is the most substantial change. The core CUDA rasterizer was re-engineered.
  - The **forward pass** no longer uses perspective projection. It now implements an **orthogonal slicing** mechanism, projecting 3D Gaussians onto a virtual plane at a given Z-coordinate. An attenuation factor based on the Gaussian's distance to the slice plane is applied to its opacity.
  - The **backward pass** has been correspondingly modified to correctly propagate gradients for position (`xyz`), opacity, scale, rotation, and intensity based on the new slicing mathematics.

> **Note**: Files ending in `* copy.*` are unmodified original source files from the 3DGS project, preserved for reference purposes only. They are not used in this project's pipeline.

## Installation

### 1. Clone the Repository
Clone this repository and its submodules recursively.
```bash
git clone https://github.com/michaelz9436/gaussian-slicing --recursive
cd gaussian-slicing
```

### 2. Set Up Conda Environment
Create and activate the Conda environment using the provided file.
```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```
*The environment name remains `gaussian_splatting` for consistency with the original project.*

## Usage Guide

### 1. Data Preparation
Place your data in a directory with the following structure:
```
./data/your_experiment_name/
├── images/             # Or the directory name you specify with -i
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── sparse.ply          # Initial sparse point cloud (can be very small)
```
The images should be named sequentially corresponding to their Z-order. The `sparse.ply` file provides initial seed points for the Gaussians.

### 2. Training
To train a new model, run the `train.py` script. The following command trains for 40,000 iterations and then automatically runs the rendering script to generate comparison images from the trained model.
```bash
python train.py -s ./data/confocal/Experiment-8144_0 -m ./output/slice_test_run --iterations 40000 && python render_slices.py -m ./output/slice_test_run
```
- `-s`: Path to the source data directory.
- `-m`: Path to the output directory where the model will be saved.
- `--iterations`: Total number of training iterations.

### 3. Hyperparameter Tuning
Key training hyperparameters can be adjusted in the `arguments/__init__.py` file within the `OptimizationParams` class.

### 4. Rendering Novel Slices
The most powerful feature is rendering novel, unseen slices at arbitrary Z-coordinates.
```bash
python render_slices.py -m ./output/slice_test_run --iteration 40000 --novel_z 21.5 22 22.5 23 23.5 24
```
- `-m`: Path to the trained model's output directory.
- `--iteration`: (Optional) Specify which iteration's checkpoint to use. Defaults to the latest.
- `--novel_z`: A list of one or more Z-coordinates at which to generate new slices.

### 5. Pruning Trained Gaussians (Optional)
After training, the model may contain many transparent or redundant Gaussians. This script can be used to filter them, reducing file size and potentially improving rendering performance.
```bash
python prune_gaussians.py ./output/slice_test_run/point_cloud/iteration_40000/point_cloud.ply \
--opacity_thresh 0.02 \
--feature_thresh 0.15 \
--scale_thresh 0.008 \
--xyz_thresh 530 \
--aspect_ratio_thresh 20
```
**Parameter Explanation**:
- **`point_cloud.ply`**: Path to the trained Gaussian model file.
- **`--opacity_thresh`**: Prunes Gaussians with an opacity below this threshold. Useful for removing near-invisible points.
- **`--feature_thresh`**: Prunes Gaussians with an intensity (feature) value below this threshold. Useful for removing very dim points.
- **`--scale_thresh`**: Prunes Gaussians whose largest scale dimension is smaller than this value. Useful for removing tiny, noisy points.
- **`--xyz_thresh`**: Prunes Gaussians whose X, Y, or Z coordinate exceeds this value. Useful for removing outliers far from the main scene.
- **`--aspect_ratio_thresh`**: Prunes Gaussians with a high aspect ratio (i.e., very elongated or flat shapes). A value of 20 means the ratio of the largest to smallest scale dimension cannot exceed 20.

### 6. Converting to Standard 3DGS Format (Optional)
This script converts the simplified Gaussian model back into the standard 3DGS format by adding placeholder values for Spherical Harmonics. This increases file size but makes the model compatible with standard 3DGS viewers.
```bash
python convert_to_standard_format.py ./output/slice_test_run/point_cloud/iteration_40000/point_cloud.ply
```

## Acknowledgements

This work is built upon the incredible research and open-source code from the authors of 3D Gaussian Splatting.

- **Project Page**: [https://repo-sam.inria.fr/fungraph/3d-gaussians/](https://repo-sam.inria.fr/fungraph/3d-gaussians/)
- **Paper**: Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics*.
- **Original Code**: `git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive`
