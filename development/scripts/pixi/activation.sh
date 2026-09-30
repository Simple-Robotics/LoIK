#! /bin/bash

if [[ $PIXI_ENVIRONMENT_PLATFORMS == *"linux"* ]];
then
  # Conda compiler is named x86_64-conda-linux-gnu-c++, ccache can't resolve it
  # (https://ccache.dev/manual/latest.html#config_compiler_type)
  export CCACHE_COMPILERTYPE=gcc
fi

# Without -isystem, some LSP can't find headers
export LOIK_CXX_FLAGS="$LOIK_CXX_FLAGS -isystem $CONDA_PREFIX/include"

# Set default build value only if not previously set
export LOIK_BUILD_TYPE=${LOIK_BUILD_TYPE:=Release}
