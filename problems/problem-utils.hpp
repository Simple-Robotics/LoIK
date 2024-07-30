#pragma once

#include "loik/fwd.hpp"

#include <pinocchio/container/boost-container-limits.hpp>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <iostream>

template < typename _Scalar >
struct DiffIKProblem {
  typedef loik::IkIdDataTypeOptimizedTpl<_Scalar> IkIdDataTypeOptimized;
  IKID_DATA_TYPEDEF_TEMPLATE(IkIdDataTypeOptimized);

	int max_iter;
  _Scalar tol_abs;
  _Scalar tol_rel;
  _Scalar tol_primal_inf;
  _Scalar tol_dual_inf;
  _Scalar tol_tail_solve;
  _Scalar rho;
  _Scalar mu;
  _Scalar mu_equality_scale_factor;

  typename IkIdDataTypeOptimized::MuUpdateStrat mu_update_strat;
  int num_eq_c = 1;
  int eq_c_dim = 6;
  bool warm_start = false;
  bool verbose = false;
  bool logging = false;

  Model robot_model;

  std::string urdf_filename;

  DVec q;

  Mat6x6 H_ref;
  Inertia H_ref_inertia;
  Motion v_ref;
  std::vector<Index> active_task_constraint_ids;
  PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) Ais;
  PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) bis;
  Scalar bound_magnitude;
  DVec lb;
  DVec ub;

}; // struct DiffIKProblem