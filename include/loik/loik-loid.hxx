//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid.hpp"

namespace loik
{

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::FwdPassInit(const DVec & q)
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    for (const auto & idx : joint_range_)
    {
      const JointModel & jmodel = model_.joints[idx];
      JointData & jdata = ik_id_data_.joints[idx];
      Index parent = model_.parents[idx];

      // computes "M"s for each joint, i.e. displacement in current joint frame caused by 'self.q_'
      jmodel.calc(jdata, q);
      ik_id_data_.liMi[idx] = model_.jointPlacements[idx] * jdata.M();
      ik_id_data_.oMi[idx] = ik_id_data_.oMi[parent] * ik_id_data_.liMi[idx];
    }

    LOIK_EIGEN_MALLOC_ALLOWED();
  } // FwdPassInit

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::FwdPass1()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();

    for (const auto & idx : joint_range_)
    {
      const JointModel & jmodel = model_.joints[idx];

      int joint_nv = jmodel.nv();
      int joint_idx_v = jmodel.idx_v();

      // build Ris and ris TODO: Ris only need to be computed once for constant mu
      ik_id_data_.Ris[idx] = mu_ineq_ * DMat::Identity(joint_nv, joint_nv);
      const auto & wi = ik_id_data_.w.segment(joint_idx_v, joint_nv);
      const auto & zi = ik_id_data_.z.segment(joint_idx_v, joint_nv);
      ik_id_data_.ris[idx].noalias() = wi - mu_ineq_ * zi;

      const Mat6x6 & H_ref = problem_.H_refs_[idx];
      const Motion & v_ref = problem_.v_refs_[idx];
      ik_id_data_.His[idx].noalias() = this->rho_ * DMat::Identity(6, 6) + H_ref;

      ik_id_data_.pis[idx].noalias() =
        -this->rho_ * ik_id_data_.vis_prev[idx].toVector() - H_ref.transpose() * v_ref.toVector();
    }

    Index c_vec_id = 0;
    for (const auto & c_id : problem_.active_task_constraint_ids_)
    {
      const Mat6x6 & Ai = problem_.Ais_[c_vec_id];
      const Vec6 & bi = problem_.bis_[c_vec_id];

      const DVec & yi = ik_id_data_.yis[c_id];
      ik_id_data_.His[c_id].noalias() += mu_eq_ * Ai.transpose() * Ai;
      ik_id_data_.pis[c_id].noalias() += Ai.transpose() * yi - mu_eq_ * Ai.transpose() * bi;

      c_vec_id++;
    }

    LOIK_EIGEN_MALLOC_ALLOWED();
  } // FwdPass1

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::BwdPass()
  {

    // loop over joint range in reverse
    for (auto it = joint_range_.rbegin(); it != joint_range_.rend(); ++it)
    {
      Index idx = *it;

      JointData & jdata = ik_id_data_.joints[idx];
      Index parent = model_.parents[idx];

      const SE3 & liMi = ik_id_data_.liMi[idx];
      const Mat6x6 & Hi = ik_id_data_.His[idx];
      const Vec6 & pi = ik_id_data_.pis[idx];
      const DMat & Si = jdata.S().matrix(); // TODO: this will cause memory allocation
      const DMat & Ri = ik_id_data_.Ris[idx];
      const DVec & ri = ik_id_data_.ris[idx];

      ik_id_data_.Dis[idx].noalias() =
        Ri + Si.transpose() * Hi * Si; // TODO: this will cause memory allocation
      ik_id_data_.Di_invs[idx].noalias() =
        ik_id_data_.Dis[idx].inverse(); // TODO: this will cause memory allocation
      const DMat & Di_inv = ik_id_data_.Di_invs[idx];
      ik_id_data_.Pis[idx].noalias() =
        DMat::Identity(6, 6)
        - Hi * Si * Di_inv * Si.transpose(); // TODO: this will cause memory allocation
      const Mat6x6 & Pi = ik_id_data_.Pis[idx];

      ik_id_data_.His[parent].noalias() +=
        liMi.toDualActionMatrix() * (Pi * Hi) * liMi.toActionMatrixInverse();
      ik_id_data_.pis[parent].noalias() +=
        liMi.toDualActionMatrix()
        * (Pi * pi - Hi * Si * Di_inv * ri); // TODO: this will cause memory allocation
    }

  } // BwdPass

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::FwdPass2()
  {

    for (const auto & idx : joint_range_)
    {

      const JointModel & jmodel = model_.joints[idx];
      JointData & jdata = ik_id_data_.joints[idx];
      Index parent = model_.parents[idx];

      int joint_nv = jmodel.nv();
      int joint_idx_v = jmodel.idx_v();

      const DMat & Di_inv = ik_id_data_.Di_invs[idx];
      const auto Si = jdata.S().matrix();
      const Mat6x6 & Hi = ik_id_data_.His[idx];
      const Vec6 & pi = ik_id_data_.pis[idx];
      const DVec & ri = ik_id_data_.ris[idx];
      const SE3 & liMi = ik_id_data_.liMi[idx];
      const Motion vi_parent = liMi.actInv(
        ik_id_data_.vis[parent]); // ith joint's parent spatial velocity in joint i's local frame

      ik_id_data_.nu.segment(joint_idx_v, joint_nv).noalias() =
        -Di_inv * (Si.transpose() * (Hi * vi_parent.toVector() + pi) + ri);

      ik_id_data_.Si_nui_s[idx] = Motion(Si * ik_id_data_.nu.segment(joint_idx_v, joint_nv));
      ik_id_data_.vis[idx] = vi_parent + ik_id_data_.Si_nui_s[idx];

      ik_id_data_.fis[idx].noalias() = Hi * ik_id_data_.vis[idx].toVector() + pi;
    }

  } // FwdPass2

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::BoxProj()
  {
    LOIK_EIGEN_MALLOC_NOT_ALLOWED();
    // update slack
    ik_id_data_.z.noalias() = problem_.ub_.cwiseMin(
      problem_.lb_.cwiseMax(ik_id_data_.nu + (1.0 / mu_ineq_) * ik_id_data_.w));
    LOIK_EIGEN_MALLOC_ALLOWED();
  } // BoxProj

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::DualUpdate()
  {

    // update dual variables associated with motion constraints 'ik_id_data_.yis'
    Index c_vec_id = 0;
    for (const auto & c_id : problem_.active_task_constraint_ids_)
    {
      const Mat6x6 & Ai = problem_.Ais_[c_vec_id];
      const Vec6 & bi = problem_.bis_[c_vec_id];
      const Motion & vi = ik_id_data_.vis[c_id];

      ik_id_data_.yis[c_id].noalias() += mu_eq_ * (Ai * vi.toVector() - bi);

      c_vec_id++;
    }

    // update dual vairables associated with inequality slack induced equality constraints
    // 'ik_id_data_.w'
    ik_id_data_.w.noalias() += mu_ineq_ * (ik_id_data_.nu - ik_id_data_.z);

  } // DualUpdate

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::UpdateQPADMMSolveLoopUtility()
  {
    // update standard qp formulation using primal dual variables from current iter
    problem_.UpdateQPADMMSolveLoop(ik_id_data_);
  } // UpdateQPADMMSolveLoopUtility

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::ComputeResiduals()
  {

    const int m = problem_.eq_c_dim_;

    // compute primal residual
    Index c_vec_id = 0;
    for (const auto & c_id : problem_.active_task_constraint_ids_)
    {
      const Mat6x6 & Ai = problem_.Ais_[c_vec_id];
      const Vec6 & bi = problem_.bis_[c_vec_id];
      const Motion & vi = ik_id_data_.vis[c_id];

      primal_residual_vec_.segment(m * (static_cast<int>(c_id) - 1), m) = Ai * vi.toVector() - bi;

      c_vec_id++;
    }

    primal_residual_vec_.segment(m * nb_, nv_) = ik_id_data_.nu - ik_id_data_.z;
    this->primal_residual_ = primal_residual_vec_.template lpNorm<Eigen::Infinity>();
    primal_residual_task_ =
      primal_residual_vec_.segment(0, m * nb_).template lpNorm<Eigen::Infinity>();
    primal_residual_slack_ =
      primal_residual_vec_.segment(m * nb_, nv_).template lpNorm<Eigen::Infinity>();

    // compute dual residual
    for (auto it = joint_range_.rbegin(); it != joint_range_.rend(); ++it)
    {
      Index idx = *it;

      const Mat6x6 & H_ref = problem_.H_refs_[idx];
      const Motion & v_ref = problem_.v_refs_[idx];
      const Motion & vi = ik_id_data_.vis[idx];
      const Vec6 & fi = ik_id_data_.fis[idx];
      const SE3 & liMi = ik_id_data_.liMi[idx];
      Index parent = model_.parents[idx];

      int row_start_idx = (static_cast<int>(idx) - 1) * 6;
      int row_start_idx_parent = 0;

      if (static_cast<int>(parent) > 0)
      {
        row_start_idx_parent = (static_cast<int>(parent) - 1) * 6;
        dual_residual_vec_.segment(row_start_idx, 6).noalias() +=
          H_ref * vi.toVector() - H_ref * v_ref.toVector() - fi;
        dual_residual_vec_.segment(row_start_idx_parent, 6).noalias() +=
          liMi.toDualActionMatrix() * fi;
      }
      else
      {
        row_start_idx_parent = 0;
        dual_residual_vec_.segment(row_start_idx, 6).noalias() +=
          H_ref * vi.toVector() - H_ref * v_ref.toVector() - fi;
      }

      const JointModel & jmodel = model_.joints[idx];
      const JointData & jdata = ik_id_data_.joints[idx];
      const int row_start_idx_wi = jmodel.idx_v();

      const auto & wi = ik_id_data_.w.segment(row_start_idx_wi, jmodel.nv());
      const auto Si = jdata.S().matrix();

      dual_residual_vec_.segment(6 * nb_ + row_start_idx_wi, jmodel.nv()).noalias() =
        Si.transpose() * fi + wi;
    }

    c_vec_id = 0;
    for (const auto & c_id : problem_.active_task_constraint_ids_)
    {
      const Mat6x6 & Ai = problem_.Ais_[c_vec_id];
      const Vec6 & yi = ik_id_data_.yis[c_id];

      int row_start_idx = (static_cast<int>(c_id) - 1) * 6;

      dual_residual_vec_.segment(row_start_idx, 6).noalias() += Ai.transpose() * yi;

      c_vec_id++;
    }

    dual_residual_vec_ = problem_.P_qp_ * problem_.x_qp_ + problem_.q_qp_
                         + problem_.A_qp_.transpose() * problem_.y_qp_;

    dual_residual_prev_ = this->dual_residual_;
    dual_residual_v_prev_ = dual_residual_v_;
    dual_residual_nu_prev_ = dual_residual_nu_;

    this->dual_residual_ = dual_residual_vec_.template lpNorm<Eigen::Infinity>();
    dual_residual_v_ = (dual_residual_vec_.segment(0, 6 * nb_)).template lpNorm<Eigen::Infinity>();
    dual_residual_nu_ =
      (dual_residual_vec_.segment(6 * nb_, nv_)).template lpNorm<Eigen::Infinity>();

    delta_dual_residual_ = this->dual_residual_ - dual_residual_prev_;
    delta_dual_residual_v_ = dual_residual_v_ - dual_residual_v_prev_;
    delta_dual_residual_nu_ = dual_residual_nu_ - dual_residual_nu_prev_;

  } // ComputeResiduals

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::CheckConvergence()
  {
    // update primal residual tolerance
    this->tol_primal_ = this->tol_abs_
                        + this->tol_rel_
                            * std::max(
                              (problem_.A_qp_ * problem_.x_qp_).template lpNorm<Eigen::Infinity>(),
                              (problem_.z_qp_).template lpNorm<Eigen::Infinity>());

    // update dual residual tolerance
    this->tol_dual_ =
      this->tol_abs_
      + this->tol_rel_
          * std::max(
            std::max(
              (problem_.P_qp_ * problem_.x_qp_).template lpNorm<Eigen::Infinity>(),
              (problem_.A_qp_.transpose() * problem_.y_qp_).template lpNorm<Eigen::Infinity>()),
            (problem_.q_qp_).template lpNorm<Eigen::Infinity>());

    // check convergence
    if ((this->primal_residual_ < this->tol_primal_) && (this->dual_residual_ < this->tol_dual_))
    {
      this->converged_ = true;

      if (this->verbose_)
      {
        std::cerr << "[FirstOrderLoik::CheckConvergence]: converged in " << this->iter_
                  << "iterations !!!" << std::endl;
      }
    }

  } // CheckConvergence

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::CheckFeasibility()
  {

    // check for primal infeasibility
    bool primal_infeasibility_cond_1 =
      (problem_.A_qp_.transpose() * problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>()
      <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();

    bool primal_infeasibility_cond_2 =
      (problem_.ub_qp_.transpose() * problem_.delta_y_qp_plus_
       + problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_)
        .value()
      <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();

    if (primal_infeasibility_cond_1 && primal_infeasibility_cond_2)
    {
      this->primal_infeasible_ = true;
      if (this->verbose_)
      {
        std::cerr
          << "WARNING [FirstOrderLoik::CheckFeasibility]: IK problem is primal infeasible !!!"
          << std::endl;
      }
    }

    // check for dual infeasibility
    bool dual_infeasibility_cond_1 =
      (problem_.P_qp_ * problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>()
      <= this->tol_dual_inf_ * (problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>();
    bool dual_infeasibility_cond_2 =
      (problem_.q_qp_.transpose() * problem_.delta_x_qp_).value()
      <= this->tol_dual_inf_ * (problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>();

    if (dual_infeasibility_cond_1 && dual_infeasibility_cond_2)
    {
      bool dual_infeasibility_cond_3 =
        ((problem_.A_qp_ * problem_.delta_x_qp_).array()
         >= -this->tol_dual_inf_ * (problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>())
          .all();
      bool dual_infeasibility_cond_4 =
        ((problem_.A_qp_ * problem_.delta_x_qp_).array()
         <= this->tol_dual_inf_ * (problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>())
          .all();

      if (dual_infeasibility_cond_3 && dual_infeasibility_cond_4)
      {
        this->dual_infeasible_ = true;
        if (this->verbose_)
        {
          std::cerr
            << "WARNING [FirstOrderLoik::CheckFeasibility]: IK problem is dual infeasible !!!"
            << std::endl;
        }
      }
    }
  } // CheckFeasibility

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template<typename _Scalar>
  void FirstOrderLoikTpl<_Scalar>::UpdateMu()
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
        "[FirstOrderLoik::UpdateMu]: mu update strategy OSQP not yet implemented"));
    }
    else if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::MAXEIGENVALUE)
    {
      // use max eigen value strategy
      throw(std::runtime_error(
        "[FirstOrderLoik::UpdateMu]: mu update strategy MAXEIGENVALUE not yet implemented"));
    }
    else
    {
      throw(std::runtime_error("[FirstOrderLoik::UpdateMu]: mu update strategy not supported"));
    }
  } // UpdateMu

} // namespace loik
