//
// Copyright (c) 2024 INRIA
//

#include <pinocchio/multibody/fwd.hpp>

#pragma once

namespace loik
{

  // IkIdDataTpl
  template<
    typename _Scalar,
    int _Options = 0,
    template<typename S, int O> class JointCollectionTpl = pinocchio::JointCollectionDefaultTpl>
  struct IkIdDataTpl;

  // IkIdDataTypeOptimizeTpl
  template<
    typename _Scalar,
    int _Options = 0,
    template<typename S, int O> class JointCollectionTpl = pinocchio::JointCollectionDefaultTpl>
  struct IkIdDataTypeOptimizedTpl;

  // IkIdSolverBaseTpl
  template<typename _Scalar>
  struct IkIdSolverBaseTpl;

  // FirstOrderLoikTpl
  template<typename _Scalar>
  struct FirstOrderLoikTpl;

  // FirstOrderLoikOptimizedTpl
  template<
    typename _Scalar,
    int _Options = 0,
    template<typename S, int O> class JointCollectionTpl = pinocchio::JointCollectionDefaultTpl>
  struct FirstOrderLoikOptimizedTpl;

  /// type alias macro
  #define IKID_DATA_TYPEDEF_GENERIC(Data,TYPENAME) \
    typedef TYPENAME pinocchio::traits<Data>::Scalar Scalar; \
    typedef TYPENAME pinocchio::traits<Data>::Model Model; \
    typedef TYPENAME pinocchio::traits<Data>::SE3 SE3; \
    typedef TYPENAME pinocchio::traits<Data>::Motion Motion; \
    typedef TYPENAME pinocchio::traits<Data>::Force Force; \
    typedef TYPENAME pinocchio::traits<Data>::Inertia Inertia; \
    typedef TYPENAME pinocchio::traits<Data>::Frame Frame; \
    typedef TYPENAME pinocchio::traits<Data>::Index Index; \
    typedef TYPENAME pinocchio::traits<Data>::JointIndex JointIndex; \
    typedef TYPENAME pinocchio::traits<Data>::FrameIndex FrameIndex; \
    typedef TYPENAME pinocchio::traits<Data>::IndexVector IndexVector; \
    typedef TYPENAME pinocchio::traits<Data>::JointModel JointModel; \
    typedef TYPENAME pinocchio::traits<Data>::JointData JointData; \
    typedef TYPENAME pinocchio::traits<Data>::JointModelVector JointModelVector; \
    typedef TYPENAME pinocchio::traits<Data>::JointDataVector JointDataVector; \
    typedef TYPENAME pinocchio::traits<Data>::DMat DMat; \
    typedef TYPENAME pinocchio::traits<Data>::DVec DVec; \
    typedef TYPENAME pinocchio::traits<Data>::Vec3 Vec3; \
    typedef TYPENAME pinocchio::traits<Data>::Vec6 Vec6; \
    typedef TYPENAME pinocchio::traits<Data>::Mat6x6 Mat6x6; \
    typedef TYPENAME pinocchio::traits<Data>::ConfigVectorType ConfigVectorType; \
    typedef TYPENAME pinocchio::traits<Data>::TangentVectorType TangentVectorType
  
  #define IKID_DATA_TYPEDEF_TEMPLATE(Data) IKID_DATA_TYPEDEF_GENERIC(Data,typename)

} // namespace loik
