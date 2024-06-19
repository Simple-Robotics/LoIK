//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-optimized.hpp"

namespace loik {
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Scalar, int _Options,
          template <typename, int> class JointCollectionTpl>
void FirstOrderLoikOptimizedTpl<_Scalar, _Options,
                                JointCollectionTpl>::ResetSolver() {

  Base::Reset(); // reset base

  tail_solve_iter_ = 0;

  primal_residual_kinematics_ = std::numeric_limits<Scalar>::infinity();
  primal_residual_task_ = std::numeric_limits<Scalar>::infinity();
  primal_residual_slack_ = std::numeric_limits<Scalar>::infinity();

  dual_residual_prev_ = std::numeric_limits<Scalar>::infinity();
  delta_dual_residual_ = std::numeric_limits<Scalar>::infinity();
  dual_residual_v_ = std::numeric_limits<Scalar>::infinity();
  ;
  dual_residual_v_prev_ = std::numeric_limits<Scalar>::infinity();
  delta_dual_residual_v_ = std::numeric_limits<Scalar>::infinity();
  dual_residual_nu_ = std::numeric_limits<Scalar>::infinity();
  dual_residual_nu_prev_ = std::numeric_limits<Scalar>::infinity();
  delta_dual_residual_nu_ = std::numeric_limits<Scalar>::infinity();

  primal_residual_vec_.setZero();
  dual_residual_vec_.setZero();

  mu_eq_ = this->mu_equality_scale_factor_ *
           this->mu_; // 'mu_' is reset to 'mu0_' during 'Base::Reset()'
  mu_ineq_ = this->mu_;

} // ResetSolver

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Scalar, int _Options,
          template <typename, int> class JointCollectionTpl>
void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::
    SolveInit(const DVec &q, const Mat6x6 &H_ref, const Motion &v_ref,
              const std::vector<Index> &active_task_constraint_ids,
              const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) & Ais,
              const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) & bis, const DVec &lb,
              const DVec &ub) {
  // reset logging if this->logging_
  if (this->logging_) {
    loik_solver_info_.Reset(); // reset logging
  }

  // reset problem description
  problem_.Reset();

  // reset IkIdData
  ik_id_data_.Reset(this->warm_start_);

  // wipe solver quantities'
  ResetSolver();

  // TODO: perform initial fwd pass, to calculate forward kinematics quantities
  FwdPassInit(q);

  // update problem formulation
  problem_.UpdateReference(H_ref, v_ref);
  problem_.UpdateIneqConstraints(lb, ub);
  problem_.UpdateEqConstraints(active_task_constraint_ids, Ais, bis);

} // SolveInit

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Scalar, int _Options,
          template <typename, int> class JointCollectionTpl>
void FirstOrderLoikOptimizedTpl<_Scalar, _Options,
                                JointCollectionTpl>::Solve() {
  // reset IkIdData recursion quantites
  ik_id_data_.ResetRecursion();

  // wipe solver quantities'
  ResetSolver();

  // solver main loop
  for (int i = 1; i < this->max_iter_; i++) {

    this->iter_ = i;

    if (this->verbose_) {
      std::cout << "===============" << std::endl;
      std::cout << "ADMM iter: " << this->iter_ << "||" << std::endl;
      std::cout << "===============" << std::endl;
    }

    loik_solver_info_.iter_list_.push_back(this->iter_);

    /// TODO: check how expensive this is vs doing the updates in main solver
    /// instead of as a single function
    ik_id_data_.UpdatePrev();

    // fwd pass 1
    FwdPass1();

    // bwd pass
    // BwdPass();
    // BwdPassOptimized();
    BwdPassOptimizedVisitor();

    // fwd pass 2
    // FwdPass2();
    // FwdPass2Optimized();
    FwdPass2OptimizedVisitor();

    // DEBUG:
    // for (Index idx : joint_range_) {
    //     std::cout << "=============================" << std::endl;
    //     std::cout << "idx: " << idx << std::endl;
    //     std::cout << "vi: " << ik_id_data_.vis[idx].toVector().transpose() <<
    //     std::endl; std::cout << "fi: " << ik_id_data_.fis[idx].transpose() <<
    //     std::endl;
    //     // std::cout << "yi: " << ik_id_data_.yis[idx].transpose() <<
    //     std::endl;
    // }

    // box projection
    BoxProj();

    // dual update
    DualUpdate();

    /// TODO: add 'UpdateQPADMMSolveLoop' equivalent to base problem formulation
    // update standard qp formulation using primal dual variables from current
    // iter problem_.UpdateQPADMMSolveLoop(ik_id_data_);

    // compute residuals
    // ComputePrimalResiduals();
    // ComputeDualResiduals();
    ComputeResiduals();

    if (this->logging_) {

      // logging residuals TODO: should be disabled for speed
      loik_solver_info_.primal_residual_task_list_.push_back(
          primal_residual_task_);
      loik_solver_info_.primal_residual_slack_list_.push_back(
          primal_residual_slack_);
      loik_solver_info_.primal_residual_list_.push_back(this->primal_residual_);
      loik_solver_info_.dual_residual_nu_list_.push_back(dual_residual_nu_);
      loik_solver_info_.dual_residual_v_list_.push_back(dual_residual_v_);
      loik_solver_info_.dual_residual_list_.push_back(this->dual_residual_);

      loik_solver_info_.mu_list_.push_back(this->mu_);
      loik_solver_info_.mu_eq_list_.push_back(mu_eq_);
      loik_solver_info_.mu_ineq_list_.push_back(mu_ineq_);
    }

    // check for convergence or infeasibility
    CheckConvergence();
    if (this->iter_ > 1) {
      CheckFeasibility();
    }

    if (this->converged_) {
      break; // converged, break out of solver main loop
    } else if (this->primal_infeasible_) {
      if (this->verbose_) {
        std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: primal "
                     "infeasibility detected at iteration: "
                  << this->iter_ << std::endl;
      }
      // problem is primal infeasible, run infeasibility tail solve
      InfeasibilityTailSolve();
      break;
    } else if (this->dual_infeasible_) {
      if (this->verbose_) {
        std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: dual "
                     "infeasibility detected at iteration: "
                  << this->iter_ << std::endl;
      }
      // problem is dual infeasibile, run infeasibility tail solve
      InfeasibilityTailSolve();
      break;
    }

    // update ADMM penalty
    UpdateMu();
  }
} // Solve

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename _Scalar, int _Options,
          template <typename, int> class JointCollectionTpl>
void FirstOrderLoikOptimizedTpl<_Scalar, _Options,
                                JointCollectionTpl>::InfeasibilityTailSolve() {
  // tail_solve_iter_ = 0;

  // while (problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() >= 1e-2 ||
  // problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() >= 1e-2) {
  //     if (this->verbose_) {
  //         if (this->iter_ >= this->max_iter_) {
  //             std::cerr << "WARNING
  //             [FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]:
  //             infeasibility detected, tail_solve exceeds max_it_," <<
  //             std::endl;
  //         }
  //     }

  //     if (this->iter_ >= this->max_iter_) {
  //         return;
  //     }

  //     this->iter_ ++ ;

  //     if (this->verbose_)
  //     {
  //         std::cout << "===============" << std::endl;
  //         std::cout << "ADMM iter: " << this->iter_ << "||" << std::endl;
  //         std::cout << "===============" << std::endl;
  //     }

  //     loik_solver_info_.iter_list_.push_back(this->iter_);

  //     tail_solve_iter_ ++;
  //     loik_solver_info_.tail_solve_iter_list_.push_back(tail_solve_iter_);

  //     ik_id_data_.UpdatePrev();

  //     // fwd pass 1
  //     FwdPass1();

  //     // bwd pass
  //     BwdPass();

  //     // fwd pass 2
  //     FwdPass2();

  //     // box projection
  //     BoxProj();

  //     // dual update
  //     DualUpdate();

  //     // update standard qp formulation using primal dual variables from
  //     current iter problem_.UpdateQPADMMSolveLoop(ik_id_data_);

  //     // compute residuals
  //     ComputeResiduals();

  //     if (this->logging_) {

  //         // logging residuals TODO: should be disabled for speed
  //         loik_solver_info_.primal_residual_task_list_.push_back(primal_residual_task_);
  //         loik_solver_info_.primal_residual_slack_list_.push_back(primal_residual_slack_);
  //         loik_solver_info_.primal_residual_list_.push_back(this->primal_residual_);
  //         loik_solver_info_.dual_residual_nu_list_.push_back(dual_residual_nu_);
  //         loik_solver_info_.dual_residual_v_list_.push_back(dual_residual_v_);
  //         loik_solver_info_.dual_residual_list_.push_back(this->dual_residual_);

  //         loik_solver_info_.mu_list_.push_back(this->mu_);
  //         loik_solver_info_.mu_eq_list_.push_back(mu_eq_);
  //         loik_solver_info_.mu_ineq_list_.push_back(mu_ineq_);

  //         loik_solver_info_.tail_solve_primal_residual_task_list_.push_back(primal_residual_task_);
  //         loik_solver_info_.tail_solve_primal_residual_slack_list_.push_back(primal_residual_slack_);
  //         loik_solver_info_.tail_solve_primal_residual_list_.push_back(this->primal_residual_);
  //         loik_solver_info_.tail_solve_dual_residual_nu_list_.push_back(dual_residual_nu_);
  //         loik_solver_info_.tail_solve_dual_residual_v_list_.push_back(dual_residual_v_);
  //         loik_solver_info_.tail_solve_dual_residual_list_.push_back(this->dual_residual_);
  //         loik_solver_info_.tail_solve_delta_x_qp_inf_norm_list_.push_back(problem_.delta_x_qp_.template
  //         lpNorm<Eigen::Infinity>());
  //         loik_solver_info_.tail_solve_delta_z_qp_inf_norm_list_.push_back(problem_.delta_z_qp_.template
  //         lpNorm<Eigen::Infinity>());

  //     }

  // }

  // if (this->verbose_) {
  //     std::cerr << "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]:
  //     tail solve completed after " << tail_solve_iter_ << " iterations." <<
  //     std::endl; std::cerr <<
  //     "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: normInf
  //     delta_x_qp_: " << problem_.delta_x_qp_.template
  //     lpNorm<Eigen::Infinity>() << std::endl; std::cerr <<
  //     "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: normInf
  //     delta_z_qp_: " << problem_.delta_z_qp_.template
  //     lpNorm<Eigen::Infinity>() << std::endl;

  // }

} // InfeasibilityTailSolve

} // namespace loik
