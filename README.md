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

User can execute the 2 pipelins by `main.py` in the `DataLab_test/` path or individually by `regi_wrappers.py` (1st pipeline) and `proc_3d_wrappers.py` (2nd pipeline) from the `DataLab_test/pcd_register/wrappers/` path. 

If you execute scipt by `main.py`, the package will choose the pipeline based on the option stored in text files in the `DataLab_test/examples` path.

# DataLab_test













