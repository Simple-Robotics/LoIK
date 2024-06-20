//
// Copyright (c) 2024 INRIA
//

#include "loik/loik-loid-data.hpp"

namespace loik
{

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTpl(const Model &);

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::IkIdDataTpl(const Model &, const int);

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::Reset(bool);

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::ResetRecursion();

  template LOIK_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void IkIdDataTpl<
    pinocchio::context::Scalar,
    pinocchio::context::Options,
    pinocchio::JointCollectionDefaultTpl>::UpdatePrev();

} // namespace loik
