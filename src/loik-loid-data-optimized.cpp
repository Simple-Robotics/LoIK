//
// Copyright (c) 2024 INRIA
//

#include "loik/loik-loid-data-optimized.hpp"

namespace loik {

template struct LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI
    IkIdDataTypeOptimizedTpl<pinocchio::context::Scalar,
                             pinocchio::context::Options,
                             pinocchio::JointCollectionDefaultTpl>;

} // namespace loik
