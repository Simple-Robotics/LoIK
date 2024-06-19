//
// Copyright (c) 2024 INRIA
//

#include "loik/loik-loid-data.hpp"

namespace loik
{

  template struct LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>;

} // namespace loik
