//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data-optimized.hpp"

namespace loik {

extern template struct LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTypeOptimizedTpl<pinocchio::context::Scalar, pinocchio::context::Options, pinocchio::JointCollectionDefaultTpl>;

} // namespace loik
