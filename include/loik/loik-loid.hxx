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
    template <typename _Scalar>
    void FirstOrderLoikTpl<_Scalar>::ResetSolver()
    {
        loik_solver_info_.Reset();        // reset logging
        problem_.Reset();                 // reset problem formulation
        Base::Reset();                    // reset base

        ik_id_data_.Reset(this->warm_start_);     // reset IkIdData

        tail_solve_iter_ = 0;

        primal_residual_kinematics_ = std::numeric_limits<Scalar>::infinity();
        primal_residual_task_ = std::numeric_limits<Scalar>::infinity();
        primal_residual_slack_ = std::numeric_limits<Scalar>::infinity();

        dual_residual_prev_ = std::numeric_limits<Scalar>::infinity();
        delta_dual_residual_ = std::numeric_limits<Scalar>::infinity();
        dual_residual_v_ = std::numeric_limits<Scalar>::infinity();;
        dual_residual_v_prev_ = std::numeric_limits<Scalar>::infinity();
        delta_dual_residual_v_ = std::numeric_limits<Scalar>::infinity();
        dual_residual_nu_ = std::numeric_limits<Scalar>::infinity();
        dual_residual_nu_prev_ = std::numeric_limits<Scalar>::infinity();
        delta_dual_residual_nu_ = std::numeric_limits<Scalar>::infinity();

        primal_residual_vec_ = 0.0 * DVec::Ones(problem_.eq_c_dim_ * nb_ + nv_);
        dual_residual_vec_ = 0.0 * DVec::Ones(6 * nb_ + nv_);
        
        mu_eq_ = this->mu_equality_scale_factor_ * this->mu_;    // 'mu_' is reset to 'mu0_' during 'Base::Reset()'
        mu_ineq_ = this->mu_;

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename _Scalar>
    void FirstOrderLoikTpl<_Scalar>::SolveInit(const DVec& q, 
                                            const Mat6x6& H_ref, const Motion& v_ref, 
                                            const std::vector<Index>& active_task_constraint_ids, const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)& Ais, const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6)& bis, 
                                            const DVec& lb, const DVec& ub)
    {
        // wipe everything, warm-start is taken care of by 'ik_id_data_.Reset()'
        ResetSolver();

        // TODO: perform initial fwd pass, to calculate forward kinematics quantities
        FwdPassInit(q);

        // update problem formulation 
        problem_.UpdateQPADMMSolveInit(H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub, model_, ik_id_data_);

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename _Scalar>
    void FirstOrderLoikTpl<_Scalar>::Solve()
    {

        // solver main loop
        for (int i = 1; i < this->max_iter_; i++) {


            this->iter_ = i;

            if (this->verbose_) 
            {
                std::cout << "===============" << std::endl;
                std::cout << "ADMM iter: " << this->iter_ << "||" << std::endl;
                std::cout << "===============" << std::endl;
            }

            loik_solver_info_.iter_list_.push_back(this->iter_);

            ik_id_data_.UpdatePrev();

            // fwd pass 1
            FwdPass1();

            // bwd pass 
            BwdPass();
            // BwdPassOptimized();

            // fwd pass 2
            FwdPass2();
            // FwdPass2Optimized();

            // DEBUG: 
            // for (Index idx : joint_range_) {
            //     std::cout << "=============================" << std::endl;
            //     std::cout << "idx: " << idx << std::endl;
            //     std::cout << "vi: " << ik_id_data_.vis[idx].toVector().transpose() << std::endl;
            //     std::cout << "fi: " << ik_id_data_.fis[idx].transpose() << std::endl;
            //     // std::cout << "yi: " << ik_id_data_.yis[idx].transpose() << std::endl;
            // }

            // box projection
            BoxProj();

            // dual update
            DualUpdate();

            // update standard qp formulation using primal dual variables from current iter
            problem_.UpdateQPADMMSolveLoop(ik_id_data_);

            // compute residuals 
            ComputeResiduals();

            if (this->logging_) {

                // logging residuals TODO: should be disabled for speed 
                loik_solver_info_.primal_residual_task_list_.push_back(primal_residual_task_);
                loik_solver_info_.primal_residual_slack_list_.push_back(primal_residual_slack_);
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
                    std::cerr << "WARNING [FirstOrderLoik::Solve]: primal infeasibility detected at iteration: " 
                              << this->iter_ 
                              << std::endl;
                }
                // problem is primal infeasible, run infeasibility tail solve
                InfeasibilityTailSolve();
                break;
            } else if (this->dual_infeasible_) {
                if (this->verbose_) {
                    std::cerr << "WARNING [FirstOrderLoik::Solve]: dual infeasibility detected at iteration: " 
                              << this->iter_ 
                              << std::endl;
                }
                // problem is dual infeasibile, run infeasibility tail solve
                InfeasibilityTailSolve();
                break;
            }

            // update ADMM penalty 
            UpdateMu();

        }
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename _Scalar>
    void FirstOrderLoikTpl<_Scalar>::Solve(const DVec& q, 
                                        const Mat6x6& H_ref, const Motion& v_ref, 
                                        const std::vector<Index>& active_task_constraint_ids, const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)& Ais, const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6)& bis, 
                                        const DVec& lb, const DVec& ub)
    {
        // wipe everything, warm-start is taken care of by 'ik_id_data_.Reset()'
        ResetSolver();

        // TODO: perform initial fwd pass, to calculate forward kinematics quantities
        FwdPassInit(q);

        // update problem formulation 
        problem_.UpdateQPADMMSolveInit(H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub, model_, ik_id_data_);

        // solver main loop
        for (int i = 1; i < this->max_iter_; i++) {


            this->iter_ = i;

            if (this->verbose_) 
            {
                std::cout << "===============" << std::endl;
                std::cout << "ADMM iter: " << this->iter_ << "||" << std::endl;
                std::cout << "===============" << std::endl;
            }

            loik_solver_info_.iter_list_.push_back(this->iter_);

            ik_id_data_.UpdatePrev();

            // fwd pass 1
            FwdPass1();

            // bwd pass 
            BwdPass();

            // fwd pass 2
            FwdPass2();

            // DEBUG: 
            // for (Index idx : joint_range_) {
            //     std::cout << "=============================" << std::endl;
            //     std::cout << "idx: " << idx << std::endl;
            //     std::cout << "vi: " << ik_id_data_.vis[idx].toVector().transpose() << std::endl;
            //     std::cout << "fi: " << ik_id_data_.fis[idx].transpose() << std::endl;
            //     // std::cout << "yi: " << ik_id_data_.yis[idx].transpose() << std::endl;
            // }

            // box projection
            BoxProj();

            // dual update
            DualUpdate();

            // update standard qp formulation using primal dual variables from current iter
            problem_.UpdateQPADMMSolveLoop(ik_id_data_);

            // compute residuals 
            ComputeResiduals();

            if (this->logging_) {

                // logging residuals TODO: should be disabled for speed 
                loik_solver_info_.primal_residual_task_list_.push_back(primal_residual_task_);
                loik_solver_info_.primal_residual_slack_list_.push_back(primal_residual_slack_);
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
                    std::cerr << "WARNING [FirstOrderLoik::Solve]: primal infeasibility detected at iteration: " 
                              << this->iter_ 
                              << std::endl;
                }
                // problem is primal infeasible, run infeasibility tail solve
                InfeasibilityTailSolve();
                break;
            } else if (this->dual_infeasible_) {
                if (this->verbose_) {
                    std::cerr << "WARNING [FirstOrderLoik::Solve]: dual infeasibility detected at iteration: " 
                              << this->iter_ 
                              << std::endl;
                }
                // problem is dual infeasibile, run infeasibility tail solve
                InfeasibilityTailSolve();
                break;
            }

            // update ADMM penalty 
            UpdateMu();

        }

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename _Scalar>
    void FirstOrderLoikTpl<_Scalar>::InfeasibilityTailSolve()
    {
        tail_solve_iter_ = 0;

        while (problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() >= 1e-2 || problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() >= 1e-2) {
            if (this->verbose_) {
                if (this->iter_ >= this->max_iter_) {
                    std::cerr << "WARNING [FirstOrderLoik::InfeasibilityTailSolve]: infeasibility detected, tail_solve exceeds max_it_," << std::endl;
                }
            }

            if (this->iter_ >= this->max_iter_) {
                return;
            }

            this->iter_ ++ ;

            if (this->verbose_) 
            {
                std::cout << "===============" << std::endl;
                std::cout << "ADMM iter: " << this->iter_ << "||" << std::endl;
                std::cout << "===============" << std::endl;
            }


            loik_solver_info_.iter_list_.push_back(this->iter_);

            tail_solve_iter_ ++;
            loik_solver_info_.tail_solve_iter_list_.push_back(tail_solve_iter_);

            ik_id_data_.UpdatePrev();

            // fwd pass 1
            FwdPass1();

            // bwd pass 
            BwdPass();

            // fwd pass 2
            FwdPass2();

            // box projection
            BoxProj();

            // dual update
            DualUpdate();

            // update standard qp formulation using primal dual variables from current iter
            problem_.UpdateQPADMMSolveLoop(ik_id_data_);

            // compute residuals 
            ComputeResiduals();

            if (this->logging_) {

                // logging residuals TODO: should be disabled for speed 
                loik_solver_info_.primal_residual_task_list_.push_back(primal_residual_task_);
                loik_solver_info_.primal_residual_slack_list_.push_back(primal_residual_slack_);
                loik_solver_info_.primal_residual_list_.push_back(this->primal_residual_);
                loik_solver_info_.dual_residual_nu_list_.push_back(dual_residual_nu_);
                loik_solver_info_.dual_residual_v_list_.push_back(dual_residual_v_);
                loik_solver_info_.dual_residual_list_.push_back(this->dual_residual_);

                loik_solver_info_.mu_list_.push_back(this->mu_);
                loik_solver_info_.mu_eq_list_.push_back(mu_eq_);
                loik_solver_info_.mu_ineq_list_.push_back(mu_ineq_);

                loik_solver_info_.tail_solve_primal_residual_task_list_.push_back(primal_residual_task_);
                loik_solver_info_.tail_solve_primal_residual_slack_list_.push_back(primal_residual_slack_);
                loik_solver_info_.tail_solve_primal_residual_list_.push_back(this->primal_residual_);
                loik_solver_info_.tail_solve_dual_residual_nu_list_.push_back(dual_residual_nu_);
                loik_solver_info_.tail_solve_dual_residual_v_list_.push_back(dual_residual_v_);
                loik_solver_info_.tail_solve_dual_residual_list_.push_back(this->dual_residual_);
                loik_solver_info_.tail_solve_delta_x_qp_inf_norm_list_.push_back(problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>());
                loik_solver_info_.tail_solve_delta_z_qp_inf_norm_list_.push_back(problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>());

            }

        }

        if (this->verbose_) {
            std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: tail solve completed after " << tail_solve_iter_ << " iterations." << std::endl;
            std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: normInf delta_x_qp_: " << problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
            std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: normInf delta_z_qp_: " << problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
            
        }

    }

} // namespace loik
