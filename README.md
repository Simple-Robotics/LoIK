# LoIK

**LoIK** is a simple yet efficient (constrained) differential inverse kinematics solver for robotics

It is designed to function as an inner solver for various downstream applications, including global inverse kinematics and sampling-based motion planning.

## Features

**LoIK** is a C++ template library, which provides:

* a set of efficient solvers for constrained differential inverse kinematics problems
* support for the [pinocchio](https://github.com/stack-of-tasks/pinocchio) rigid-body dynamics library
* an interface to the [IKBench](https://github.com/Simple-Robotics/IKBench) inverse kinematics benchmark library which can be used to compare different IK solver performances
* Python bindings leveraging [eigenpy](https://github.com/stack-of-tasks/eigenpy) {Next release}

To cite **LoIK** in your publications, software, and research articles.
Please refer to the [Citation section](#citing-loik) for further details.

## Installation

<!-- ### From Conda

From either conda-forge or [our channel](https://anaconda.org/simple-robotics/loik).

```bash
conda install -c conda-forge loik  # or -c conda-forge
``` -->


### Build from source

```bash
git clone https://github.com/Simple-Robotics/LoIK --recursive
cd LoIK
mkdir build && cd build
cmake .. -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=your_install_folder -DCMAKE_CXX_FLAGS="-march=native"
make -jNCPUS
make install
```

#### Dependencies
* [Eigen3](https://eigen.tuxfamily.org) >= 3.4.0
* [Boost](https://www.boost.org) >= 1.84.0
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio)>=3.0.0 | [conda](https://anaconda.org/conda-forge/pinocchio)
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.4.0 | [conda](https://anaconda.org/conda-forge/eigenpy) (Python bindings)
* (optional) [example-robot-data](https://github.com/Gepetto/example-robot-data)>=4.1.0 | [conda](https://anaconda.org/conda-forge/example-robot-data) (required for examples and benchmarks)
* a C++17 compliant compiler

#### Notes

* For developers, add the `-D CMAKE_EXPORT_COMPILE_COMMANDS=1` when working with language servers e.g. clangd.
* To check for runtime Eigen memory allocation, add `-D CHECK_RUNTIME_MALLOC=ON`
* By default, building the library will instantiate the templates for the `double` scalar type.
* To build against a Conda environment, activate the environment and run `export CMAKE_PREFIX_PATH=$CONDA_PREFIX` before running CMake and use `$CONDA_PREFIX` as your install folder, i.e. add flag `-D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX`.

### Build/install from source with Pixi

To build **LoIK** from source the easiest way is to use [Pixi](https://pixi.sh/latest/#installation).

[Pixi](https://pixi.sh/latest/) is a cross-platform package management tool for developers that
will install all required dependencies in `.pixi` directory.
It's used by our CI agent so you have the guarantee to get the right dependencies.

Run the following command to install dependencies, configure, build and test the project:

```bash
pixi run test
```

The project will be built in the `build` directory.
You can now run `pixi shell` and build the project with `cmake` and `ninja` manually.


## Benchmarking

We recommend [Flame Graphs](https://github.com/brendangregg/FlameGraph) for performance analysis.
Please refer to this code analysis [tutorial](https://github.com/Simple-Robotics/code-analysis-tools?tab=readme-ov-file#install-1) for installation and usage of flame graph.

## Citing LoIK

To cite **LoIK**, please use the following bibtex entry:

```bibtex
@misc{loikapi,
  author = {Wingo, Bruce and Vaillant, Joris and Sathya, Ajay and Caron, Stéphane and Carpentier, Justin},
  title = {LoIK},
  url = {https://github.com/Simple-Robotics/LoIK}
}
```
Please also consider citing the reference paper for the **LoIK** algorithm:

```bibtex
@inproceedings{wingoLoIK2024,
  title = {{Linear-time Differential Inverse Kinematics: an Augmented Lagrangian Perspective}},
  author = {Wingo, Bruce and Sathya, Ajay and Caron, Stéphane and Hutchinson, Seth and Carpentier, Justin},
  year = {2024},
  booktitle={Robotics: Science and Systems},
  note = {https://inria.hal.science/hal-04607809v1}
}
```

## Contributors

* [Bruce Wingo](https://bwingo47.github.io/) (Inria, Georgia Tech): main developer and manager of the project
* [Ajay Sathya](https://scholar.google.com/citations?user=A00LDswAAAAJ&hl=en) (Inria): algorithm developer and core developer. 
* [Joris Vaillant](https://github.com/jorisv) (Inria): core developer
* [Stéphane Caron](https://scaron.info/) (Inria): core developer
* [Seth Hutchinson](https://faculty.cc.gatech.edu/~seth/) (Georgia Tech): project instructor
* [Justin Carpentier](https://jcarpent.github.io/) (Inria): project instructor and core developer

## Acknowledgments

The development of **LoIK** is actively supported by the [Willow team](https://www.di.ens.fr/willow/) at [@INRIA](http://www.inria.fr) Paris.
