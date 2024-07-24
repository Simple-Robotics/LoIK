//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data.hpp"

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
  IkIdDataTpl<_Scalar, _Options, JointCollectionTpl>::IkIdDataTpl(const Model & model)
  : IkIdDataTpl(model, 6) // delegating constructor call
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
  IkIdDataTpl<_Scalar, _Options, JointCollectionTpl>::IkIdDataTpl(
    const Model & model, const int eq_c_dim)
  : joints(0)
  , oMi(static_cast<std::size_t>(model.njoints), SE3::Identity())
  , liMi(static_cast<std::size_t>(model.njoints), SE3::Identity())
  , nu(DVec::Zero(model.nv))
  , nu_prev(DVec::Zero(model.nv))
  , vis(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , Si_nui_s(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , ovis(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , vis_prev(static_cast<std::size_t>(model.njoints), Motion::Zero())
  , His_aba(static_cast<std::size_t>(model.njoints), Mat6x6::Identity())
  , His(static_cast<std::size_t>(model.njoints), Mat6x6::Identity())
  , Pis(static_cast<std::size_t>(model.njoints), Mat6x6::Identity())
  , pis(static_cast<std::size_t>(model.njoints), Vec6::Zero())
  , Dis(static_cast<std::size_t>(model.njoints))
  , Di_invs(static_cast<std::size_t>(model.njoints))
  , Ris(static_cast<std::size_t>(model.njoints))
  , ris(static_cast<std::size_t>(model.njoints))
  , fis(static_cast<std::size_t>(model.njoints), Vec6::Zero())
  , yis(static_cast<std::size_t>(model.njoints), DVec::Zero(eq_c_dim))
  , yis_prev(static_cast<std::size_t>(model.njoints), DVec::Zero(eq_c_dim))
  , w(DVec::Zero(model.nv))
  , w_prev(DVec::Zero(model.nv))
  , z(DVec::Zero(model.nv))
  , z_prev(DVec::Zero(model.nv))
  , eq_c_dim(eq_c_dim)
  {
    /* Create data structure associated to the joints */
    for (JointIndex i = 0; i < static_cast<JointIndex>(model.njoints); ++i)
    {
      // create joint datas
      joints.push_back(
        pinocchio::CreateJointData<_Scalar, _Options, JointCollectionTpl>::run(model.joints[i]));

      // fill Dis, Di_invs, Ris, and ris
      Ris[i] = DMat::Zero(model.joints[i].nv(), model.joints[i].nv());
      ris[i] = DVec::Zero(model.joints[i].nv());
      Dis[i] = DMat::Zero(model.joints[i].nv(), model.joints[i].nv());
      Di_invs[i] = DMat::Zero(model.joints[i].nv(), model.joints[i].nv());
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
  /// \brief Reset all quantites
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTpl<_Scalar, _Options, JointCollectionTpl>::Reset(const bool warm_start)
  {

    if (!warm_start)
    {
      // no warm start, wipe everything
      nu.setZero();
      nu_prev.setZero();
      w.setZero();
      w_prev.setZero();
      z.setZero();
      z_prev.setZero();

      for (auto & idx : joint_full_range)
      {
        oMi[idx].setIdentity();
        liMi[idx].setIdentity();

        vis[idx].setZero();
        Si_nui_s[idx].setZero();
        ovis[idx].setZero();
        vis_prev[idx].setZero();

        His_aba[idx].setZero();
        His[idx].setZero();
        Pis[idx].setZero();
        pis[idx].setZero();
        Dis[idx].setZero();
        Di_invs[idx].setZero();
        Ris[idx].setZero();
        ris[idx].setZero();
        fis[idx].setZero();
        yis[idx].setZero();
        yis_prev[idx].setZero();
      }
    }
    else
    {
      // warm start, keep primal and dual variables from previous solve() call
      nu_prev = nu;
      w_prev = w;
      z_prev = z;

      for (auto & idx : joint_full_range)
      {
        oMi[idx].setIdentity();
        liMi[idx].setIdentity();

        // wipe recursion related quantites
        His_aba[idx].setZero();
        His[idx].setZero();
        Pis[idx].setZero();
        pis[idx].setZero();
        Dis[idx].setZero();
        Di_invs[idx].setZero();
        Ris[idx].setZero();
        ris[idx].setZero();
        fis[idx].setZero();

        // update _prev variables without wiping current primal dual variables
        vis_prev[idx] = vis[idx];
        yis_prev[idx] = yis[idx];
      }
    }

  } // Reset

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Reset all recursion related quantites
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTpl<_Scalar, _Options, JointCollectionTpl>::ResetRecursion()
  {

    // no warm start, wipe everything
    nu.setZero();
    nu_prev.setZero();
    w.setZero();
    w_prev.setZero();
    z.setZero();
    z_prev.setZero();

    for (auto & idx : joint_full_range)
    {
      vis[idx].setZero();
      Si_nui_s[idx].setZero();
      ovis[idx].setZero();
      vis_prev[idx].setZero();

      His_aba[idx].setZero();
      His[idx].setZero();
      Pis[idx].setZero();
      pis[idx].setZero();
      Dis[idx].setZero();
      Di_invs[idx].setZero();
      Ris[idx].setZero();
      ris[idx].setZero();
      fis[idx].setZero();
      yis[idx].setZero();
      yis_prev[idx].setZero();
    }

  } // ResetRecursion

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief update _prev primal dual variables with that of current values
  ///
  template<typename _Scalar, int _Options, template<typename S, int> class JointCollectionTpl>
  void IkIdDataTpl<_Scalar, _Options, JointCollectionTpl>::UpdatePrev()
  {
    nu_prev = nu;
    for (auto & idx : joint_full_range)
    {
      vis_prev[idx] = vis[idx];
      yis_prev[idx] = yis[idx];
    }
    w_prev = w;
    z_prev = z;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief operator "==" overload, as a Non-member function for symmetric behavior, i.e. a == b
  ///        and b == a shouldm give the same result.
  ///
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator==(
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_1,
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_2)
  {
    bool value =
      ik_id_data_1.joints == ik_id_data_2.joints && ik_id_data_1.oMi == ik_id_data_2.oMi
      && ik_id_data_1.liMi == ik_id_data_2.liMi && ik_id_data_1.nu == ik_id_data_2.nu
      && ik_id_data_1.nu_prev == ik_id_data_2.nu_prev && ik_id_data_1.vis == ik_id_data_2.vis
      && ik_id_data_1.Si_nui_s == ik_id_data_2.Si_nui_s && ik_id_data_1.ovis == ik_id_data_2.ovis
      && ik_id_data_1.vis_prev == ik_id_data_2.vis_prev
      && ik_id_data_1.His_aba == ik_id_data_2.His_aba && ik_id_data_1.His == ik_id_data_2.His
      && ik_id_data_1.Pis == ik_id_data_2.Pis && ik_id_data_1.pis == ik_id_data_2.pis
      && ik_id_data_1.Dis == ik_id_data_2.Dis && ik_id_data_1.Di_invs == ik_id_data_2.Di_invs
      && ik_id_data_1.Ris == ik_id_data_2.Ris && ik_id_data_1.ris == ik_id_data_2.ris
      && ik_id_data_1.fis == ik_id_data_2.fis && ik_id_data_1.yis == ik_id_data_2.yis
      && ik_id_data_1.yis_prev == ik_id_data_2.yis_prev && ik_id_data_1.w == ik_id_data_2.w
      && ik_id_data_1.w_prev == ik_id_data_2.w_prev && ik_id_data_1.z == ik_id_data_2.z
      && ik_id_data_1.z_prev == ik_id_data_2.z_prev
      && ik_id_data_1.joint_full_range == ik_id_data_2.joint_full_range
      && ik_id_data_1.joint_range == ik_id_data_2.joint_range;

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
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_1,
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data_2)
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
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data,
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
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data,
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data)
  {
    return CompareDataIkIdData(ik_id_data, data);
  }

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator==(
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data)
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
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data,
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data)
  {
    return !CompareDataIkIdData(ik_id_data, data);
  }

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  bool operator!=(
    const pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const IkIdDataTpl<Scalar, Options, JointCollectionTpl> & ik_id_data)
  {
    return !CompareDataIkIdData(ik_id_data, data);
  }

} // namespace loik
