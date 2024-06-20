//
// Copyright (c) 2024 INRIA
//

#include "loik/loik-loid-data-optimized.hpp"

namespace loik
{

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTypeOptimizedTpl(const Model &);

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTypeOptimizedTpl(const Model &, const int);

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::Reset(bool);

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::ResetRecursion();

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void IkIdDataTypeOptimizedTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::UpdatePrev();

} // namespace loik
