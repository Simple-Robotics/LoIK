//
// Copyright (c) 2024 INRIA
//

#include <pinocchio/multibody/fwd.hpp>

#pragma once 

namespace loik {

    // IkIdDataTpl
    template<typename _Scalar, int _Options = 0, template<typename S, int O> class JointCollectionTpl = pinocchio::JointCollectionDefaultTpl>
    struct IkIdDataTpl;

    // IkIdDataTypeOptimizeTpl
    template<typename _Scalar, int _Options = 0, template<typename S, int O> class JointCollectionTpl = pinocchio::JointCollectionDefaultTpl>
    struct IkIdDataTypeOptimizedTpl;

    // IkIdSolverBaseTpl
    template<typename _Scalar>
    struct IkIdSolverBaseTpl;

    // FirstOrderLoikTpl
    template<typename _Scalar>
    struct FirstOrderLoikTpl;

    // FirstOrderLoikOptimizedTpl
    template<typename _Scalar, int _Options = 0, template<typename S, int O> class JointCollectionTpl = pinocchio::JointCollectionDefaultTpl>
    struct FirstOrderLoikOptimizedTpl;
    




} // namespace loik