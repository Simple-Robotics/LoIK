//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data-optimized.hpp"

#include <pinocchio/multibody/liegroup/liegroup-algo.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/fwd.hpp>
#include <pinocchio/utils/string-generator.hpp>

#include <numeric>

namespace loik
{

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief IkIdDataTpl constructor with default constraint dimension of 6
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  IkIdDataTypeOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::IkIdDataTypeOptimizedTpl(
    const Model & model)
  : IkIdDataTypeOptimizedTpl(model, 1) // delegating constructor call
  {
    //
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief IkIdDataTpl constructor
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  IkIdDataTypeOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::IkIdDataTypeOptimizedTpl(
    const Model & model, const int num_eq_c)
  : joints(0)
  , oMi(static_cast<std::size_t>(model.njoints), SE3::Identity())
  , liMi(static_cast<std::size_t>(model.njoints), SE3::Identity())
  , nu(DVec::Zero(model.nv))
  , nu_prev(DVec::Zero(model.nv))
  , vis(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , ovis(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , vis_prev(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , His(static_cast<std::size_t>(model.njoints), Mat6x6::Identity())
  , His_aba(static_cast<std::size_t>(model.njoints), Mat6x6::Identity())
  , pis(static_cast<std::size_t>(model.njoints), Force::Zero())
  , pis_aba(static_cast<std::size_t>(model.njoints), Force::Zero())
  , R(DVec::Zero(model.nv))
  , r(DVec::Zero(model.nv))
  , fis(static_cast<std::size_t>(model.njoints), Force::Zero())
  , delta_fis(static_cast<std::size_t>(model.njoints), Force::Zero())
  , yis(static_cast<std::size_t>(num_eq_c), Vec6::Zero())
  , delta_yis(static_cast<std::size_t>(num_eq_c), Vec6::Zero())
  , w(DVec::Zero(model.nv))
  , delta_w(DVec::Zero(model.nv))
  , z(DVec::Zero(model.nv))
  , z_prev(DVec::Zero(model.nv))
  , num_eq_c(num_eq_c)
  , Aty(static_cast<std::size_t>(num_eq_c), Vec6::Zero())
  , Aty_prev(static_cast<std::size_t>(num_eq_c), Vec6::Zero())
  , fis_diff_plus_Aty(static_cast<std::size_t>(model.njoints), Force::Zero())
  , delta_fis_diff_plus_Aty(static_cast<std::size_t>(model.njoints), Force::Zero())
  , Href_v(static_cast<std::size_t>(model.njoints), Vec6::Zero())
  , Av_minus_b(static_cast<std::size_t>(num_eq_c), Vec6::Zero())
  , Stf_plus_w(DVec::Zero(model.nv))
  , delta_Stf_plus_w(DVec::Zero(model.nv))
  , bT_delta_y_plus(0.0)
  , bT_delta_y_minus(0.0)
  , Av_inf_norm(0.0)
  , nu_inf_norm(0.0)
  , Href_v_inf_norm(0.0)
  , fis_diff_plus_Aty_inf_norm(0.0)
  , Stf_plus_w_inf_norm(0.0)
  , delta_fis_diff_plus_Aty_inf_norm(0.0)
  , delta_Stf_plus_w_inf_norm(0.0)
  , delta_vis_inf_norm(0.0)
  , delta_nu_inf_norm(0.0)
  , delta_z_inf_norm(0.0)
  , delta_fis_inf_norm(0.0)
  , delta_yis_inf_norm(0.0)
  , delta_w_inf_norm(0.0)
  {
    /* Create data structure associated to the joints */
    for (JointIndex i = 0; i < static_cast<JointIndex>(model.njoints); ++i)
    {
      // create joint datas
      joints.push_back(
        pinocchio::CreateJointData<_Scalar, _Options, JointCollectionTpl>::run(model.joints[i]));
    }

    // initialize utility members
    joint_full_range.reserve(static_cast<std::size_t>(model.njoints));
    joint_full_range.resize(static_cast<std::size_t>(model.njoints));
    joint_range.reserve(static_cast<std::size_t>(model.njoints) - 1);
    joint_range.resize(static_cast<std::size_t>(model.njoints) - 1);
    std::iota(joint_full_range.begin(), joint_full_range.end(), 0); // [0, nb_ - 1]
    std::iota(joint_range.begin(), joint_range.end(), 1);           // [1, nb_ - 1]
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief reset all quantities, called by solver during SolveInit(), only called once per problem
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTypeOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::Reset(const bool warm_start)
  {

    if (!warm_start)
    {
      w.setZero();
      z.setZero();
      nu.setZero();
      for (auto & idx : joint_full_range)
      {
        vis[idx].setZero();
        fis[idx].setZero();
        fis_diff_plus_Aty[idx].setZero();
      }
    }
  } // Reset

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief reset only recursion related quantities, called once at the begining of Solve(), not
  ///        resetting oMi and liMi, i.e. assuming robot state hasn't changed
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTypeOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::ResetRecursion()
  {
    w.setZero();
    z.setZero();

    for (auto & idx : joint_full_range)
    {
      vis[idx].setZero();
      fis[idx].setZero();
      fis_diff_plus_Aty[idx].setZero();
    }

    for (int idx = 0; idx < num_eq_c; idx++)
    {
      yis[idx].setZero();
      Aty[idx].setZero();
    }

  } // ResetRecursion

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief reset only inf norm related quantities, called at the begining of each solver inner
  ///        iterations
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTypeOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::ResetInfNorms()
  {
    bT_delta_y_plus = 0.0;
    bT_delta_y_minus = 0.0;
    Av_inf_norm = 0.0;
    nu_inf_norm = 0.0;
    Href_v_inf_norm = 0.0;
    fis_diff_plus_Aty_inf_norm = 0.0;
    Stf_plus_w_inf_norm = 0.0;
    delta_fis_diff_plus_Aty_inf_norm = 0.0;
    delta_Stf_plus_w_inf_norm = 0.0;
    delta_vis_inf_norm = 0.0;
    delta_nu_inf_norm = 0.0;
    delta_z_inf_norm = 0.0;
    delta_fis_inf_norm = 0.0;
    delta_yis_inf_norm = 0.0;
    delta_w_inf_norm = 0.0;
  } // ResetInfNorms

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief update _prev primal dual variables with that of current values
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTypeOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::UpdatePrev()
  {
    vis_prev = vis;
    nu_prev = nu;
    z_prev = z;
  } // UpdatePrev

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief operator "==" overload, as a Non-member function for symmetric behavior, i.e. a == b
  ///        and b == a shouldm give the same result.
  ///
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator==(
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_1,
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_2)
  {
    bool value =
      ik_id_data_1.joints == ik_id_data_2.joints && ik_id_data_1.oMi == ik_id_data_2.oMi
      && ik_id_data_1.liMi == ik_id_data_2.liMi && ik_id_data_1.nu == ik_id_data_2.nu
      && ik_id_data_1.nu_prev == ik_id_data_2.nu_prev && ik_id_data_1.vis == ik_id_data_2.vis
      && ik_id_data_1.ovis == ik_id_data_2.ovis && ik_id_data_1.vis_prev == ik_id_data_2.vis_prev
      && ik_id_data_1.His_aba == ik_id_data_2.His_aba && ik_id_data_1.His == ik_id_data_2.His
      && ik_id_data_1.pis == ik_id_data_2.pis && ik_id_data_1.pis_aba == ik_id_data_2.pis_aba
      && ik_id_data_1.R == ik_id_data_2.R && ik_id_data_1.r == ik_id_data_2.r
      && ik_id_data_1.fis == ik_id_data_2.fis && ik_id_data_1.delta_fis == ik_id_data_2.delta_fis
      && ik_id_data_1.yis == ik_id_data_2.yis && ik_id_data_1.delta_yis == ik_id_data_2.delta_yis
      && ik_id_data_1.w == ik_id_data_2.w && ik_id_data_1.delta_w == ik_id_data_2.delta_w
      && ik_id_data_1.z == ik_id_data_2.z && ik_id_data_1.z_prev == ik_id_data_2.z_prev
      && ik_id_data_1.joint_full_range == ik_id_data_2.joint_full_range
      && ik_id_data_1.joint_range == ik_id_data_2.joint_range
      && ik_id_data_1.Aty == ik_id_data_2.Aty && ik_id_data_1.Aty_prev == ik_id_data_2.Aty_prev
      && ik_id_data_1.fis_diff_plus_Aty == ik_id_data_2.fis_diff_plus_Aty
      && ik_id_data_1.delta_fis_diff_plus_Aty == ik_id_data_2.delta_fis_diff_plus_Aty
      && ik_id_data_1.Href_v == ik_id_data_2.Href_v
      && ik_id_data_1.Av_minus_b == ik_id_data_2.Av_minus_b
      && ik_id_data_1.Stf_plus_w == ik_id_data_2.Stf_plus_w
      && ik_id_data_1.delta_Stf_plus_w == ik_id_data_2.delta_Stf_plus_w
      && ik_id_data_1.bT_delta_y_plus == ik_id_data_2.bT_delta_y_plus
      && ik_id_data_1.bT_delta_y_minus == ik_id_data_2.minus
      && ik_id_data_1.Av_inf_norm == ik_id_data_2.Av_inf_norm
      && ik_id_data_1.nu_inf_norm == ik_id_data_2.nu_inf_norm
      && ik_id_data_1.Href_v_inf_norm == ik_id_data_2.Href_v_inf_norm
      && ik_id_data_1.fis_diff_plus_Aty_inf_norm == ik_id_data_2.fis_diff_plus_Aty_inf_norm
      && ik_id_data_1.Stf_plus_w_inf_norm == ik_id_data_2.Stf_plus_w_inf_norm
      && ik_id_data_1.delta_fis_diff_plus_Aty_inf_norm
           == ik_id_data_2.delta_fis_diff_plus_Aty_inf_norm
      && ik_id_data_1.delta_Stf_plus_w_inf_norm == ik_id_data_2.delta_Stf_plus_w_inf_norm
      && ik_id_data_1.delta_vis_inf_norm == ik_id_data_2.delta_vis_inf_norm
      && ik_id_data_1.delta_nu_inf_norm == ik_id_data_2.delta_nu_inf_norm
      && ik_id_data_1.delta_z_inf_norm == ik_id_data_2.delta_z_inf_norm
      && ik_id_data_1.delta_fis_inf_norm == ik_id_data_2.delta_fis_inf_norm
      && ik_id_data_1.delta_yis_inf_norm == ik_id_data_2.delta_yis_inf_norm
      && ik_id_data_1.delta_w_inf_norm == ik_id_data_2.delta_w_inf_norm;

    return value;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief operator "!=" overload, as a Non-member function for symmetric behavior, i.e. a != b
  ///        and b != a shouldm give the same result.
  ///
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator!=(
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_1,
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_2)
  {
    return !(ik_id_data_1 == ik_id_data_2);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief utility function for defining symmetric operator "==" overload between IkIdData and
  /// Data
  ///
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool CompareDataIkIdData(
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data,
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data)
  {
    bool value = ik_id_data.oMi == data.oMi && ik_id_data.liMi == data.liMi;

    // check joint data created btw "ik_id_data" and "data" are the same
    for (std::size_t j = 1; j < static_cast<std::size_t>(ik_id_data.joints.size()); ++j)
    {
      value &= ik_id_data.joints.at(j) == data.joints.at(j);
    }

    return value;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief symmetric operator "==" overloads between IkIdData and Data
  ///
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator==(
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data,
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data)
  {
    return CompareDataIkIdData(ik_id_data, data);
  }

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator==(
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data)
  {
    return CompareDataIkIdData(ik_id_data, data);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief symmetric operator "!=" overloads between IkIdData and Data
  ///
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator!=(
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data,
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data)
  {
    return !CompareDataIkIdData(ik_id_data, data);
  }

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator!=(
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> & ik_id_data)
  {
    return !CompareDataIkIdData(ik_id_data, data);
  }

} // namespace loik
