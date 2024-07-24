//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data-optimized.hpp"

namespace loik
{

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTypeOptimizedTpl(const Model &);

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTypeOptimizedTpl(const Model &, const int);

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::Reset(bool);

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::ResetRecursion();

  extern template LOIK_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::UpdatePrev();

} // namespace loik
