# Instruction on technical interview (Research engineer)

1. A candidate receives this repository from DataLabs.
2. The candidate does its task (its detail is written below) in this repository.
3. The candidate submits it to DataLabs.
4. The DataLabs and the candidate have an interview.

## Task of candidate

The candidate chooses one of the programming languages (Python or C++), and does the task below.

### Python

1. Read and understand [Open3D tutorial for pointcloud registration pipeline](http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html) and [3D data processing](./3d_processing.md), which include the following 2 pipelines respectively:

- 1st pipeline:
  - load pointcloud
  - registrations:
    - preprocess
      - downsample
      - compute features (normal/FPFH)
    - rough registration (with RANSAC)
    - fine registration (with ICP)

<!-- This line is necessary to split a list into two -->

- 2nd pipeline:
  - load pointcloud
  - [3D data processing](./3d_processing.md)
    - Compute approximate Curvature
    - Projection of points to the plane

2. Implement the 2 pipelines above in the following modules:

   - a reusable python module in `pcd_register/`
   - its unittest in `tests/`
   - application code which imports methods/classes from `pcd_register` in `main.py`

   In doing it, be careful on the following points:

   - Make `git` commit with appropriate granularity
   - write [docstring](https://realpython.com/documenting-python-code/)
   - write [type annotation](https://realpython.com/python-type-checking/)
   - Follow [PEP8](https://www.python.org/dev/peps/pep-0008/)
   - If you use external library, specify all the libraries and their version in one of the following files:
     - `pyproject.toml/poetry.lock`
     - `requirements.txt`

3. Write the necessary documents to this `README.md`, which includes:

   - how to install
   - how to run the code

For the format, see the end of this file.

### C++

_To be written..._

## Task of interviewer(s)

1. Once the candidate submits the "homework", the interviewer(s) reviews it.
2. By the date of interview, prepare list of questions/discussion topics.

## Time budget

The candidate is asked to submit her/his homeworks as soon as it is done, and up to three weeks.

## What to be reviewed by the technical interview

### From submitted repository

- Quality of code

### From interview

- Whether the candidate can explain intension of his/her design/implementation.
- Whether the candidate can constructive discussion on how to improve design/implementation

## Technical references

### software engineering

- [git [realpython]](https://realpython.com/python-git-github-intro/)
- [what is unittest? [wikipedia]](https://en.wikipedia.org/wiki/Unit_testing)

### python

- [realpython](https://realpython.com/)
- python module
- installing/managing python environment:
  - [pip [realpython]](https://realpython.com/what-is-pip/)
  - [poetry](https://python-poetry.org/)
- unittest tool ([pytest](https://docs.pytest.org/en/6.2.x/))

### C++

_To be written..._

### 3D data

- [Open3D](http://www.open3d.org/docs/release/index.html)

---

**Below the candidate should edit to make proper README for his/her codes.**

## Prerequisites

Before you begin, ensure you have met the following requirements:

### OS

This package is developed from the `macOS` version `12.7.5`. In the future, it needs to be tested for working on other OSs.

### Other software dependencies

In order to run this package, the below packages needed to be installed or added in the `pyproject.toml`.

- `poetry`: A tool for dependency management and packaging in Python.
- `python>=3.10,<3.11`: A general Python package.
- `numpy<2`: Fundamental package for array computing in Python.
- `h5py^3.11.0`: Read and write HDF5 files from Python.
- `click^8.1.7`: Composable command line interface toolkit.
- `open3d^0.18.0`: A Modern Library for 3D Data Processing.
- `tqdm^4.66.5`: Fast, Extensible Progress Meter.

## Installation

The environment of the package is managed by the `poetry`. So, the first user needs to install the `poetry`.

```
pip install poetry
```

Second, users need to download this package from GitHub.

```
git clone git@github.com:MyoungchulK/DataLab_test.git
```

At last, go to the `DataLab_test` and execute the `poetry install` to install the software dependencies.

```
cd DataLab_test/
poetry install
```

## Usage

Follow these steps:

User can execute the 2 pipelines by `main.py` in the `DataLab_test/` path or individually by `regi_wrappers.py` (1st pipeline) and `proc_3d_wrappers.py` (2nd pipeline) from the `DataLab_test/pcd_register/wrappers/` path. Each script in the `wrappers/` path will call the necessary classes in the `DataLab_test/pcd_register/tools/` path to do the calculation.
 
If you execute the script by `main.py`, the package will choose the pipeline based on the option stored in text files in the `DataLab_test/examples` path.

The specific script option for the pipeline is managed by adding the variables to the text file in dictionary format. The script will read the dictionary using the JSON package and use it to control the pipeline. All available variables are laned inside of the text file.

In the `DataLab_test/examples` path, the `regi_var_ex.txt` stores variables for the registration pipeline. The `proc_3d_var_ex.txt` is storing the 3d process variables. User can modify each text file for their analysis. If the User doesn't specify the input pcd file, the script will automatically use the [ICP dataset](https://www.open3d.org/docs/release/python_api/open3d.data.DemoICPPointClouds.html#open3d.data.DemoICPPointClouds).

Users can use the text file by the `-v` or `--dat_var` option at the terminal.

At the `DataLab_test/` path:
```
python3 main.py -v examples/regi_var_ex.txt # For the registration pipeline
python3 main.py -v examples/proc_3d_var_ex.txt # For the 3d process pipeline
```

At the `DataLab_test/pcd_register/wrappers/` path:
```
python3 regi_wrappers.py -v examples/regi_var_ex.txt # For the registration pipeline
python3 proc_3d_wrappers.py -v examples/proc_3d_var_ex.txt # For the 3d process pipeline
```

The results will be saved in the `DataLab_test/outputs` paths as a default output path. If the user specifies the output path, a default path will be ignored. The results will be saved in three different formats. The point cloud data will be saved in `pcd` format. The secondary information that is not `pcd` format will be saved in `h5` format. The plot about registration will also saved in `png` format in the path. If the user sets the `use_debug` variable in the text file to the `True`, the script also saves the middle step of the calculation for more information.

The below variables are the information that saved in the `h5` format.

From registration pipelie:

- `src_fpfh`: The fpfh fearture of the source pcd file.
- `tar_fpfh`: The fpfh fearture of the target pcd file.
- `voxels`: The voxel size for the down sampling of the both pcd files. 
- `radius`: The radius for the down sampling and the fpfh fearture of the both pcd files.
- `max_nns`: The KDTree variables for the down sampling and the fpfh fearture of the both pcd files.
- `bbox_points`: The edge point of the bounding box for the both pcd files. 
- `bbox_min_max`: The pcd point that touchs the bounding box for the both pcd files. 
- `trans_mtx_pre`: The transformation matrix that used during the preprocess.
- `trans_mtx_ransac`: The transformation matrix that calculated from the RANSAC registration.
- `fit_ransac`: The fitness variable that calculated from the RANSAC registration.
- `rmse_ransac`: The RMSE variable that calculated from the RANSAC registration.
- `corr_ransac`: The correspondence set that calculated from the RANSAC registration.
- `trans_mtx_icp`: The transformation matrix that calculated from the ICP registration.
- `fit_icp`: The fitness variable that calculated from the ICP registration.
- `rmse_icp`: The RMSE variable that calculated from the ICP registration.
- `corr_icp`: The correspondence set that calculated from the ICP registration.

From 3d process pipelie:

- `covar_mtx`: The covariance matrix at a point with an index `i`.
- `approx_curv`: The approximate curvature at the point `i`.
- `proj_pts`: The project points within the radius at a point with index `i` to the plane.
- `centriod`: The centriod of the radius `r` at a point `i`.
- `eigen_val`: The eigen value at the point `i`.
- `eigen_vec`: The eigen vector at the point `i`.
- `nomal_vec`: The normal vector at the point `i`.
- `displace_vec`: The displacement vector between the point `i` and neighboting points.
- `pts_i`: The point with an index `i`.
- `pts_nei`: The neighboring points within the radius `r` at a point `i`.

# DataLab_test













