# Build and develop with pixi

The easiest way to set up a development environment is to use [pixi](https://pixi.sh/latest/#installation).

[pixi](https://pixi.sh/latest/) is a cross-platform package manager for developers.
It installs all required dependencies in the `.pixi` directory.
It's used by our CI, so you get the same stable and tested dependencies.

Run the following command to install dependencies, configure, build and test the project:

```bash
pixi run test
```

The project is built in the `build` directory.

The typical workflow is:

```bash
pixi shell
pixi run configure
ninja -C build
```

After `pixi run configure`, use `cmake` and `ninja` manually to reconfigure and build the project.

## Environments

The pixi manifest contains many environments. The most common ones are:

- **default**: core LoIK

To activate a specific environment, run:

```bash
pixi shell -e default
```
