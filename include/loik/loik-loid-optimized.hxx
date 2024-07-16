//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-optimized.hpp"

namespace loik
{
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename _Scalar, int _Options, template<typename,int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::ResetSolver()
  {
      
      Base::Reset();                    // reset base

      tail_solve_iter_ = 0;


      delta_y_qp_inf_norm_ = 0.0;
      A_qp_T_delta_y_qp_inf_norm_ = 0.0;
      ub_qp_T_delta_y_qp_plus_ = 0.0;
      lb_qp_T_delta_y_qp_minus_ = 0.0;
      primal_infeasibility_cond_1_ = false;
      primal_infeasibility_cond_2_ = false;

      mu_eq_ = this->mu_equality_scale_factor_ * this->mu_;    // 'mu_' is reset to 'mu0_' during 'Base::Reset()'
      mu_ineq_ = this->mu_;

  } // ResetSolver


  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename _Scalar, int _Options, template<typename,int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::SolveInit(const DVec& q, 
                                                    const Mat6x6& H_ref, const Motion& v_ref, 
                                                    const std::vector<Index>& active_task_constraint_ids, const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)& Ais, const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6)& bis, 
                                                    const DVec& lb, const DVec& ub)
  {
      // reset logging if this->logging_
      if (this->logging_) {
          loik_solver_info_.Reset();        // reset logging
      }

      // reset problem description
      problem_.Reset();               

      // reset IkIdData
      ik_id_data_.Reset(this->warm_start_);

      // wipe solver quantities'
      ResetSolver();

      

      // update problem formulation 
      problem_.UpdateReference(H_ref, v_ref);
      problem_.UpdateIneqConstraints(lb, ub);
      problem_.UpdateEqConstraints(active_task_constraint_ids, Ais, bis);

      FwdPassInit(q);

  } // SolveInit


  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename _Scalar, int _Options, template<typename,int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::Solve()
  {
      // reset IkIdData recursion quantites
      ik_id_data_.ResetRecursion();

      // wipe solver quantities'
      ResetSolver();

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

          ik_id_data_.ResetInfNorms();

          // fwd pass 1
          FwdPass1();

          // bwd pass 
          BwdPassOptimizedVisitor();

          // fwd pass 2
          FwdPass2OptimizedVisitor();

          // box projection
          BoxProj();

          // dual update
          DualUpdate();


          ComputeResiduals();

          if (this->logging_) {

              // logging residuals, should be disabled for speed 
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
          // CheckFeasibility();
          if (this->iter_ > 1) {
              CheckFeasibility();
          }

          if (this->converged_) {
              break; // converged, break out of solver main loop
          } else if (this->primal_infeasible_) {
              if (this->verbose_) {
                  std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: primal infeasibility detected at iteration: " 
                            << this->iter_ 
                            << std::endl;
              }
              // problem is primal infeasible, run infeasibility tail solve
              InfeasibilityTailSolve();
              break;
          } else if (this->dual_infeasible_) {
              if (this->verbose_) {
                  std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: dual infeasibility detected at iteration: " 
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
  } // Solve




  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename _Scalar, int _Options, template<typename,int> class JointCollectionTpl>
  void FirstOrderLoikOptimizedTpl<_Scalar, _Options, JointCollectionTpl>::InfeasibilityTailSolve()
  {
      
  } // InfeasibilityTailSolve


} // namespace loik
