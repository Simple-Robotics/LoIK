//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data.hpp"

namespace loik
{

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTpl(const Model &);

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTpl(const Model &, const int);

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::Reset(bool);

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::ResetRecursion();

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::UpdatePrev();

} // namespace loik
