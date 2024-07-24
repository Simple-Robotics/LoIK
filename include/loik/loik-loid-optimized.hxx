//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-optimized.hpp"

namespace loik
{

  namespace internal
  {

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
    struct LoikBackwardStepVisitor
    : public pinocchio::fusion::JointUnaryVisitorBase<
        LoikBackwardStepVisitor<Scalar, Options, JointCollectionTpl>>
    {
      typedef pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> Model;
      typedef IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> IkIdData;

      typedef boost::fusion::vector<const Model &, IkIdData &> ArgsType;

      template<typename JointModel>
      static void algo(
        const pinocchio::JointModelBase<JointModel> & jmodel,
        pinocchio::JointDataBase<typename JointModel::JointDataDerived> & jdata,
        const Model & model,
        IkIdData & ik_id_data)
      {
        LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        using JointIndex = typename Model::JointIndex;
        using Index = typename pinocchio::Index;
        using Force = typename IkIdData::Force;
        using SE3 = typename IkIdData::SE3;
        using Mat6x6 = typename IkIdData::Mat6x6;
        using DVec = typename IkIdData::DVec;
        using Vec6 = typename IkIdData::Vec6;

        JointIndex idx = jmodel.id();
        Index parent = model.parents[idx];

        const SE3 & liMi = ik_id_data.liMi[idx];
        Mat6x6 & Hi_aba = ik_id_data.His_aba[idx];
        const Mat6x6 & Hi = ik_id_data.His[idx];
        const Force & pi = ik_id_data.pis[idx];
        Force & pi_aba = ik_id_data.pis_aba[idx];

        const DVec & R = ik_id_data.R;
        DVec & r = ik_id_data.r;

        jmodel.calc_aba(jdata.derived(), jmodel.jointVelocitySelector(R), Hi_aba, parent > 0);

        ik_id_data.His_aba[parent].noalias() +=
          pinocchio::impl::internal::SE3actOn<Scalar>::run(liMi, Hi_aba);
        ik_id_data.His[parent].noalias() = ik_id_data.His_aba[parent];

        jmodel.jointVelocitySelector(r) += jdata.S().transpose() * pi;
        const Vec6 & tmp_expr = jdata.UDinv() * jmodel.jointVelocitySelector(r);
        pi_aba.linear() -= tmp_expr.template segment<3>(Force::LINEAR);
        pi_aba.angular() -= tmp_expr.template segment<3>(Force::ANGULAR);
        ik_id_data.pis[parent] += liMi.act(pi_aba);
        ik_id_data.pis_aba[parent] = ik_id_data.pis[parent];

        LOIK_EIGEN_MALLOC_ALLOWED();

      } // LoikBackwardStepVisitor::algo()

    }; // struct LoikBackwardStepVisitor

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
    struct LoikForwardStep2Visitor
    : public pinocchio::fusion::JointUnaryVisitorBase<
        LoikForwardStep2Visitor<Scalar, Options, JointCollectionTpl>>
    {
      typedef pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> Model;
      typedef IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> IkIdData;
      typedef IkProblemFormulationOptimized<Scalar> ProblemFormulation;

      typedef boost::fusion::vector<const Model &, IkIdData &, const ProblemFormulation &> ArgsType;

      template<typename JointModel>
      static void algo(
        const pinocchio::JointModelBase<JointModel> & jmodel,
        pinocchio::JointDataBase<typename JointModel::JointDataDerived> & jdata,
        const Model & model,
        IkIdData & ik_id_data,
        const ProblemFormulation & problem)
      {

        LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        using JointIndex = typename Model::JointIndex;
        using Index = typename pinocchio::Index;
        using Motion = typename IkIdData::Motion;
        using Force = typename IkIdData::Force;
        using SE3 = typename IkIdData::SE3;
        using Mat6x6 = typename IkIdData::Mat6x6;

        JointIndex idx = jmodel.id();
        Index parent = model.parents[idx];

        const Mat6x6 & Hi = ik_id_data.His[idx];
        const Force & pi = ik_id_data.pis[idx];
        const SE3 & liMi = ik_id_data.liMi[idx];

        Motion vi_parent = liMi.actInv(
          ik_id_data.vis[parent]); // ith joint's parent spatial velocity in joint i's local frame

        jmodel.jointVelocitySelector(ik_id_data.nu) =
          -jdata.UDinv().transpose() * vi_parent.toVector()
          - jdata.Dinv() * jmodel.jointVelocitySelector(ik_id_data.r);

        if (
          jmodel.jointVelocitySelector(ik_id_data.nu).template lpNorm<Eigen::Infinity>()
          > ik_id_data.nu_inf_norm)
        {
          ik_id_data.nu_inf_norm =
            jmodel.jointVelocitySelector(ik_id_data.nu).template lpNorm<Eigen::Infinity>();
        }

        ik_id_data.vis[idx] = vi_parent;
        ik_id_data.vis[idx] += jdata.S() * jmodel.jointVelocitySelector(ik_id_data.nu);

        ik_id_data.delta_fis[idx] = ik_id_data.fis[idx]; // copy `fis` before update

        ik_id_data.fis[idx].linear() =
          (Hi * ik_id_data.vis[idx].toVector()).template segment<3>(Force::LINEAR) + pi.linear();
        ik_id_data.fis[idx].angular() =
          (Hi * ik_id_data.vis[idx].toVector()).template segment<3>(Force::ANGULAR) + pi.angular();

        ik_id_data.delta_fis[idx] = ik_id_data.fis[idx] - ik_id_data.delta_fis[idx];

        if (
          (ik_id_data.delta_fis[idx].toVector()).template lpNorm<Eigen::Infinity>()
          > ik_id_data.delta_fis_inf_norm)
        {
          ik_id_data.delta_fis_inf_norm =
            (ik_id_data.delta_fis[idx].toVector()).template lpNorm<Eigen::Infinity>();
        }

        // update Href_v
        ik_id_data.Href_v[idx].noalias() = problem.H_refs_[idx] * ik_id_data.vis[idx].toVector();

        if (ik_id_data.Href_v[idx].template lpNorm<Eigen::Infinity>() > ik_id_data.Href_v_inf_norm)
        {
          ik_id_data.Href_v_inf_norm = ik_id_data.Href_v[idx].template lpNorm<Eigen::Infinity>();
        }

        // update delta_vis_inf_norm
        if (
          (ik_id_data.vis[idx] - ik_id_data.vis_prev[idx])
            .toVector()
            .template lpNorm<Eigen::Infinity>()
          > ik_id_data.delta_vis_inf_norm)
        {
          ik_id_data.delta_vis_inf_norm = (ik_id_data.vis[idx] - ik_id_data.vis_prev[idx])
                                            .toVector()
                                            .template lpNorm<Eigen::Infinity>();
        }

        LOIK_EIGEN_MALLOC_ALLOWED();

      } // LoikForwardStep2Visitor::algo

    }; // struct LoikForwardStep2Visitor

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
    struct LoikBackwardStep2Visitor
    : public pinocchio::fusion::JointUnaryVisitorBase<
        LoikBackwardStep2Visitor<Scalar, Options, JointCollectionTpl>>
    {
      typedef pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> Model;
      typedef IkIdDataTypeOptimizedTpl<Scalar, Options, JointCollectionTpl> IkIdData;
      typedef IkProblemFormulationOptimized<Scalar> ProblemFormulation;
      typedef typename IkIdData::DVec ResidualVec;

      typedef boost::fusion::
        vector<const Model &, IkIdData &, const ProblemFormulation &, ResidualVec &>
          ArgsType;

      template<typename JointModel>
      static void algo(
        const pinocchio::JointModelBase<JointModel> & jmodel,
        pinocchio::JointDataBase<typename JointModel::JointDataDerived> & jdata,
        const Model & model,
        IkIdData & ik_id_data,
        const ProblemFormulation & problem,
        ResidualVec & dual_residual_vec)
      {
        LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        using JointIndex = typename Model::JointIndex;
        using Index = typename pinocchio::Index;
        using Force = typename IkIdData::Force;
        using SE3 = typename IkIdData::SE3;
        using DVec = typename IkIdData::DVec;
        using Vec6 = typename IkIdData::Vec6;

        JointIndex idx = jmodel.id();
        Index parent = model.parents[idx];
        const SE3 & liMi = ik_id_data.liMi[idx];
        const Force & fi = ik_id_data.fis[idx];
        const DVec & w = ik_id_data.w;
        const Vec6 & Href_v = ik_id_data.Href_v[idx];
        const Vec6 & Href_vref = problem.Hv[idx];

        // update fis_diff_plus_Aty
        ik_id_data.fis_diff_plus_Aty[idx] += -fi;

        ik_id_data.fis_diff_plus_Aty[parent] += liMi.act(fi);

        // update delta_fis_diff_plus_Aty
        ik_id_data.delta_fis_diff_plus_Aty[idx] =
          ik_id_data.fis_diff_plus_Aty[idx] - ik_id_data.delta_fis_diff_plus_Aty[idx];

        // update delta_fis_diff_plus_Aty_inf_norm
        if (
          (ik_id_data.delta_fis_diff_plus_Aty[idx].toVector()).template lpNorm<Eigen::Infinity>()
          > ik_id_data.delta_fis_diff_plus_Aty_inf_norm)
        {
          ik_id_data.delta_fis_diff_plus_Aty_inf_norm =
            (ik_id_data.delta_fis_diff_plus_Aty[idx].toVector()).template lpNorm<Eigen::Infinity>();
        }

        // update fis_diff_plus_Aty_inf_norm
        if (
          (ik_id_data.fis_diff_plus_Aty[idx]).toVector().template lpNorm<Eigen::Infinity>()
          > ik_id_data.fis_diff_plus_Aty_inf_norm)
        {
          ik_id_data.fis_diff_plus_Aty_inf_norm =
            (ik_id_data.fis_diff_plus_Aty[idx]).toVector().template lpNorm<Eigen::Infinity>();
        }

        // update dual residual vector
        dual_residual_vec.template segment<6>((static_cast<int>(idx) - 1) * 6).noalias() =
          Href_v - Href_vref + ik_id_data.fis_diff_plus_Aty[idx].toVector();

        // update Stf_plus_w
        jmodel.jointVelocitySelector(ik_id_data.Stf_plus_w) =
          jdata.S().transpose() * fi + jmodel.jointVelocitySelector(w);

        // update Stf_plus_w_inf_norm
        if (
          (jmodel.jointVelocitySelector(ik_id_data.Stf_plus_w)).template lpNorm<Eigen::Infinity>()
          > ik_id_data.Stf_plus_w_inf_norm)
        {
          ik_id_data.Stf_plus_w_inf_norm = (jmodel.jointVelocitySelector(ik_id_data.Stf_plus_w))
                                             .template lpNorm<Eigen::Infinity>();
        }

        LOIK_EIGEN_MALLOC_ALLOWED();

      } // LoikBackwardStep2Visitor::algo

    }; // struct LoikBackwardStep2Visitor

  } // namespace internal

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void
  FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::FwdPassInit(const DVec & q)
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    for (const auto & idx : joint_range_)
    {
      JointModel & jmodel = model_.joints[idx];
      JointData & jdata = ik_id_data_.joints[idx];
      Index parent = model_.parents[idx];

      // computes "M"s for each joint, i.e. displacement in current joint frame caused by 'self.q_'
      jmodel.calc(jdata, q);
      ik_id_data_.liMi[idx] = model_.jointPlacements[idx] * jdata.M();
      ik_id_data_.oMi[idx] = ik_id_data_.oMi[parent] * ik_id_data_.liMi[idx];
    }

    // to properlly handle warm-starting:
    if (!warm_start_)
    {
      // no warm_start
      Index c_vec_id = 0;
      for (const auto & c_id : problem_.active_task_constraint_ids_)
      {
        ik_id_data_.yis[c_vec_id].setZero();
        ik_id_data_.Aty[c_vec_id].setZero();
        c_vec_id++;
      }
    }

    LOIK_EIGEN_MALLOC_ALLOWED();

  } // FwdPassInit

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::FwdPass1()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    ik_id_data_.R.setOnes();
    ik_id_data_.R *= mu_ineq_;
    ik_id_data_.r.noalias() = ik_id_data_.w - mu_ineq_ * ik_id_data_.z;

    for (const auto & idx : joint_range_)
    {
      JointModel & jmodel = model_.joints[idx];

      const Mat6x6 & H_ref = problem_.H_refs_[idx];
      const Vec6 & Hv_i = problem_.Hv[idx];

      ik_id_data_.His[idx].setIdentity();
      ik_id_data_.His[idx] *= this->rho_;
      ik_id_data_.His[idx].noalias() += H_ref;

      ik_id_data_.His_aba[idx].noalias() = ik_id_data_.His[idx];

      ik_id_data_.pis[idx].linear() = -this->rho_ * ik_id_data_.vis_prev[idx].linear();
      ik_id_data_.pis[idx].angular() = -this->rho_ * ik_id_data_.vis_prev[idx].angular();
      ik_id_data_.pis[idx].linear() -= Hv_i.template segment<3>(Motion::LINEAR);
      ik_id_data_.pis[idx].angular() -= Hv_i.template segment<3>(Motion::ANGULAR);

      ik_id_data_.pis_aba[idx] = ik_id_data_.pis[idx];
    }

    Index c_vec_id = 0;
    for (const auto & c_id : problem_.active_task_constraint_ids_)
    {

      const Mat6x6 & AtA_i = problem_.AtA[c_vec_id];
      const Vec6 & Atb_i = problem_.Atb[c_vec_id];
      const Vec6 & Aty_i = ik_id_data_.Aty[c_vec_id];

      ik_id_data_.His[c_id].noalias() += mu_eq_ * AtA_i;
      ik_id_data_.His_aba[c_id].noalias() += mu_eq_ * AtA_i;
      ik_id_data_.pis[c_id].linear() += (Aty_i - mu_eq_ * Atb_i).template segment<3>(Force::LINEAR);
      ik_id_data_.pis[c_id].angular() +=
        (Aty_i - mu_eq_ * Atb_i).template segment<3>(Force::ANGULAR);
      ik_id_data_.pis_aba[c_id] = ik_id_data_.pis[c_id];

      c_vec_id++;
    }

    LOIK_EIGEN_MALLOC_ALLOWED();

  } // FwdPass1

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::BwdPassOptimizedVisitor()
  {
    typedef internal::LoikBackwardStepVisitor<_Scalar, _Options, JointCollectionTpl> loik_bwd_pass;
    for (Index i = (Index)model_.njoints - 1; i > 0; --i)
    {
      loik_bwd_pass::run(
        model_.joints[i], ik_id_data_.joints[i],
        typename loik_bwd_pass::ArgsType(model_, ik_id_data_));
    }

  } // BwdPassOptimizedVisitor

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::FwdPass2OptimizedVisitor()
  {
    typedef internal::LoikForwardStep2Visitor<_Scalar, _Options, JointCollectionTpl> loik_fwd_pass2;
    ik_id_data_.delta_fis_diff_plus_Aty = ik_id_data_.fis_diff_plus_Aty;
    for (Index i = 1; i < (Index)model_.njoints; ++i)
    {
      loik_fwd_pass2::run(
        model_.joints[i], ik_id_data_.joints[i],
        typename loik_fwd_pass2::ArgsType(model_, ik_id_data_, problem_));

      ik_id_data_.fis_diff_plus_Aty[i]
        .setZero(); // this line is needed because in `BwdPass2`, `fis_diff_plus_Aty` for all
                    // indeces are updated with `+=` but in `DualUpdate`, only indeces where a
                    // constraint is defined will `fis_diff_plus_Aty` be updated with `=`
    }

    // update `delta_nu_inf_norm`
    ik_id_data_.delta_nu_inf_norm =
      (ik_id_data_.nu - ik_id_data_.nu_prev).template lpNorm<Eigen::Infinity>();

  } // FwdPass2OptimizedVisitor

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::BoxProj()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();
    // update slack
    ik_id_data_.z.noalias() = problem_.ub_.cwiseMin(
      problem_.lb_.cwiseMax(ik_id_data_.nu + (1.0 / mu_ineq_) * ik_id_data_.w));

    // update delta_z_inf_norm
    ik_id_data_.delta_z_inf_norm =
      (ik_id_data_.z - ik_id_data_.z_prev).template lpNorm<Eigen::Infinity>();

    // update primal residual vector bottom half
    primal_residual_vec_.segment(6 * nb_, nv_).noalias() = ik_id_data_.nu - ik_id_data_.z;

    LOIK_EIGEN_MALLOC_ALLOWED();
  } // BoxProj

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::DualUpdate()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    // update dual variables associated with motion constraints 'ik_id_data_.yis'
    Index c_vec_id = 0;
    for (const auto & c_id : problem_.active_task_constraint_ids_)
    {
      const Mat6x6 & Ai = problem_.Ais_[c_vec_id];
      const Vec6 & bi = problem_.bis_[c_vec_id];
      const Motion & vi = ik_id_data_.vis[c_id];

      // update ik_id_data_.Av_minus_b
      ik_id_data_.Av_minus_b[c_vec_id].noalias() = Ai * vi.toVector() - bi;

      // update ik_id_data_.delta_yis
      ik_id_data_.delta_yis[c_vec_id].noalias() = mu_eq_ * ik_id_data_.Av_minus_b[c_vec_id];

      // update ik_id_data_.yis
      ik_id_data_.yis[c_vec_id].noalias() += ik_id_data_.delta_yis[c_vec_id];

      // update ik_id_data_.Aty_i
      ik_id_data_.Aty[c_vec_id].noalias() = Ai.transpose() * ik_id_data_.yis[c_vec_id];

      // update ik_id_data_.delta_yis_inf_norm
      if (
        ik_id_data_.delta_yis[c_vec_id].template lpNorm<Eigen::Infinity>()
        > ik_id_data_.delta_yis_inf_norm)
      {
        ik_id_data_.delta_yis_inf_norm =
          ik_id_data_.delta_yis[c_vec_id].template lpNorm<Eigen::Infinity>();
      }

      // update primal residual top half
      primal_residual_vec_.template segment<6>(6 * (static_cast<int>(c_id) - 1)).noalias() =
        ik_id_data_.Av_minus_b[c_vec_id];

      // initial assignment of delta_fis_diff_plus_Aty to fis_diff_plus_Aty

      // initialize dual residual vector segment corresponding to c_id
      ik_id_data_.fis_diff_plus_Aty[c_id].linear().noalias() =
        ik_id_data_.Aty[c_vec_id].template segment<3>(Force::LINEAR);
      ik_id_data_.fis_diff_plus_Aty[c_id].angular().noalias() =
        ik_id_data_.Aty[c_vec_id].template segment<3>(Force::ANGULAR);

      // update 'bT_delta_y_plus' and 'bT_delta_y_minus'
      ik_id_data_.bT_delta_y_plus +=
        (bi.transpose() * ik_id_data_.delta_yis[c_vec_id].cwiseMax(0))[0];
      ik_id_data_.bT_delta_y_minus +=
        (bi.transpose() * ik_id_data_.delta_yis[c_vec_id].cwiseMin(0))[0];

      // update Av_inf_norm
      if ((Ai * vi.toVector()).template lpNorm<Eigen::Infinity>() > ik_id_data_.Av_inf_norm)
      {
        ik_id_data_.Av_inf_norm = (Ai * vi.toVector()).template lpNorm<Eigen::Infinity>();
      }

      c_vec_id++;
    }

    // update dual vairables associated with inequality slack induced equality constraints
    // 'ik_id_data_.w'
    ik_id_data_.delta_w.noalias() = mu_ineq_ * (ik_id_data_.nu - ik_id_data_.z);

    ik_id_data_.w.noalias() += ik_id_data_.delta_w;

    ik_id_data_.delta_w_inf_norm = ik_id_data_.delta_w.template lpNorm<Eigen::Infinity>();

    LOIK_EIGEN_MALLOC_ALLOWED();
  } // DualUpdate

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::BwdPass2OptimizedVisitor()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();
    ik_id_data_.delta_Stf_plus_w.noalias() = ik_id_data_.Stf_plus_w;
    LOIK_EIGEN_MALLOC_ALLOWED();

    typedef internal::LoikBackwardStep2Visitor<_Scalar, _Options, JointCollectionTpl>
      loik_bwd_pass2;
    for (Index i = (Index)model_.njoints - 1; i > 0; --i)
    {
      loik_bwd_pass2::run(
        model_.joints[i], ik_id_data_.joints[i],
        typename loik_bwd_pass2::ArgsType(model_, ik_id_data_, problem_, dual_residual_vec_));
    }

    LOIK_EIGEN_MALLOC_NOT_ALLOWED();
    ik_id_data_.delta_Stf_plus_w.noalias() = ik_id_data_.Stf_plus_w - ik_id_data_.delta_Stf_plus_w;
    ik_id_data_.delta_Stf_plus_w_inf_norm =
      ik_id_data_.delta_Stf_plus_w.template lpNorm<Eigen::Infinity>();
    dual_residual_vec_.segment(6 * nb_, nv_).noalias() = ik_id_data_.Stf_plus_w;
    LOIK_EIGEN_MALLOC_ALLOWED();

  } // BwdPass2OptimizedVisitor

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::ComputePrimalResiduals()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    this->primal_residual_ = primal_residual_vec_.template lpNorm<Eigen::Infinity>();
    primal_residual_task_ =
      primal_residual_vec_.segment(0, 6 * nb_).template lpNorm<Eigen::Infinity>();
    primal_residual_slack_ =
      primal_residual_vec_.segment(6 * nb_, nv_).template lpNorm<Eigen::Infinity>();

    LOIK_EIGEN_MALLOC_ALLOWED();
  } // ComputePrimalResiduals

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::ComputeDualResiduals()
  {
    BwdPass2OptimizedVisitor();

    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    this->dual_residual_ = dual_residual_vec_.template lpNorm<Eigen::Infinity>();
    dual_residual_v_ = (dual_residual_vec_.segment(0, 6 * nb_)).template lpNorm<Eigen::Infinity>();
    dual_residual_nu_ =
      (dual_residual_vec_.segment(6 * nb_, nv_)).template lpNorm<Eigen::Infinity>();

    LOIK_EIGEN_MALLOC_ALLOWED();
  } // ComputeDualResiduals

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::ComputeResiduals()
  {
    ComputePrimalResiduals();
    ComputeDualResiduals();
  } // ComputeResiduals

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::CheckConvergence()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    this->tol_primal_ = this->tol_abs_
                        + this->tol_rel_
                            * std::max(
                              std::max(ik_id_data_.Av_inf_norm, ik_id_data_.nu_inf_norm),
                              std::max(problem_.bis_inf_norm_, ik_id_data_.nu_inf_norm));

    this->tol_dual_ =
      this->tol_abs_
      + this->tol_rel_
          * std::max(
            std::max(
              ik_id_data_.Href_v_inf_norm,
              std::max(ik_id_data_.fis_diff_plus_Aty_inf_norm, ik_id_data_.Stf_plus_w_inf_norm)),
            problem_.Hv_inf_norm_);

    // check convergence
    if ((this->primal_residual_ < this->tol_primal_) && (this->dual_residual_ < this->tol_dual_))
    {
      this->converged_ = true;

      if (this->verbose_)
      {
        std::cerr << "[FirstOrderLoikOptimizedTpl::CheckConvergence]: converged in " << this->iter_
                  << "iterations !!!" << std::endl;
      }
    }

    LOIK_EIGEN_MALLOC_ALLOWED();

  } // CheckConvergence

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::CheckFeasibility()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();
    // check for primal infeasibility
    delta_y_qp_inf_norm_ = std::max(
      ik_id_data_.delta_fis_inf_norm,
      std::max(ik_id_data_.delta_yis_inf_norm, ik_id_data_.delta_w_inf_norm));

    A_qp_T_delta_y_qp_inf_norm_ =
      std::max(ik_id_data_.delta_fis_diff_plus_Aty_inf_norm, ik_id_data_.delta_Stf_plus_w_inf_norm);

    primal_infeasibility_cond_1_ =
      A_qp_T_delta_y_qp_inf_norm_ <= this->tol_primal_inf_ * delta_y_qp_inf_norm_;

    ub_qp_T_delta_y_qp_plus_ = ik_id_data_.bT_delta_y_plus;
    ub_qp_T_delta_y_qp_plus_ += (problem_.ub_.transpose() * ik_id_data_.delta_w.cwiseMax(0))[0];
    lb_qp_T_delta_y_qp_minus_ = ik_id_data_.bT_delta_y_minus;
    lb_qp_T_delta_y_qp_minus_ += (problem_.lb_.transpose() * ik_id_data_.delta_w.cwiseMin(0))[0];

    primal_infeasibility_cond_2_ = (ub_qp_T_delta_y_qp_plus_ + lb_qp_T_delta_y_qp_minus_)
                                   <= this->tol_primal_inf_ * delta_y_qp_inf_norm_;

    if (primal_infeasibility_cond_1_ && primal_infeasibility_cond_2_)
    {
      this->primal_infeasible_ = true;
      if (this->verbose_)
      {
        std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::CheckFeasibility]: IK problem is primal "
                     "infeasible !!!"
                  << std::endl;
      }
    }

    delta_x_qp_inf_norm_ = std::max(ik_id_data_.delta_vis_inf_norm, ik_id_data_.delta_nu_inf_norm);

    LOIK_EIGEN_MALLOC_ALLOWED();

  } // CheckFeasibility

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::UpdateMu()
  {
    if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::DEFAULT)
    {
      // update mu by threasholding primal and dual residual ratio
      if (this->primal_residual_ > 10 * this->dual_residual_)
      {
        this->mu_ *= 10;

        mu_eq_ = this->mu_equality_scale_factor_ * this->mu_;
        mu_ineq_ = this->mu_;
        return;
      }
      else if (this->dual_residual_ > 10 * this->primal_residual_)
      {
        this->mu_ *= 0.1;

        mu_eq_ = this->mu_equality_scale_factor_ * this->mu_;
        mu_ineq_ = this->mu_;
        return;
      }
      else
      {
        return;
      }
    }
    else if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::OSQP)
    {
      // using OSQP strategy
      throw(std::runtime_error(
        "[FirstOrderLoikOptimizedTpl::UpdateMu]: mu update strategy OSQP not yet implemented"));
    }
    else if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::MAXEIGENVALUE)
    {
      // use max eigen value strategy
      throw(std::runtime_error("[FirstOrderLoikOptimizedTpl::UpdateMu]: mu update strategy "
                               "MAXEIGENVALUE not yet implemented"));
    }
    else
    {
      throw(std::runtime_error(
        "[FirstOrderLoikOptimizedTpl::UpdateMu]: mu update strategy not supported"));
    }
  } // UpdateMu

} // namespace loik
