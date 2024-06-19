//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data.hpp"

namespace loik
{

  extern template struct LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>;

} // namespace loik
