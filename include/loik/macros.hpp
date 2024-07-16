//
// Copyright (c) 2024 INRIA
//

#pragma once

#ifdef LOIK_EIGEN_CHECK_MALLOC
  #define LOIK_EIGEN_ALLOW_MALLOC(allowed) ::Eigen::internal::set_is_malloc_allowed(allowed)
  #define LOIK_EIGEN_MALLOC_ALLOWED() LOIK_EIGEN_ALLOW_MALLOC(true)
  #define LOIK_EIGEN_MALLOC_NOT_ALLOWED() LOIK_EIGEN_ALLOW_MALLOC(false)
#else
  #define LOIK_EIGEN_ALLOW_MALLOC(allowed)
  #define LOIK_EIGEN_MALLOC_ALLOWED()
  #define LOIK_EIGEN_MALLOC_NOT_ALLOWED()
#endif
