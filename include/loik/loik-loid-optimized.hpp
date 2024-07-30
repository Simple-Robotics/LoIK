//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "fwd.hpp"
#include "loik/macros.hpp"
#include "loik/ik-id-description-optimized.hpp"
#include "loik/loik-loid-data-optimized.hpp"
#include "loik/task-solver-base.hpp"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/visitor.hpp>

namespace loik
{

  template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
  struct FirstOrderLoikOptimizedTpl : IkIdSolverBaseTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef IkIdSolverBaseTpl<_Scalar> Base;
    using ProblemFormulation = IkProblemFormulationOptimized<_Scalar>;
    using IkIdData = IkIdDataTypeOptimizedTpl<_Scalar>;
    IKID_DATA_TYPEDEF_TEMPLATE(IkIdData);
    

    struct LoikSolverInfo : Base::SolverInfo
    {
      explicit LoikSolverInfo(const int max_iter)
      : Base::SolverInfo(max_iter)
      {
        primal_residual_kinematics_list_.reserve(static_cast<std::size_t>(max_iter));
        primal_residual_task_list_.reserve(static_cast<std::size_t>(max_iter));
        primal_residual_slack_list_.reserve(static_cast<std::size_t>(max_iter));

        dual_residual_v_list_.reserve(static_cast<std::size_t>(max_iter));
        dual_residual_nu_list_.reserve(static_cast<std::size_t>(max_iter));

        mu_eq_list_.reserve(static_cast<std::size_t>(max_iter));
        mu_ineq_list_.reserve(static_cast<std::size_t>(max_iter));

        tail_solve_iter_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_primal_residual_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_primal_residual_kinematics_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_primal_residual_task_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_primal_residual_slack_list_.reserve(static_cast<std::size_t>(max_iter));

        tail_solve_dual_residual_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_dual_residual_v_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_dual_residual_nu_list_.reserve(static_cast<std::size_t>(max_iter));

        tail_solve_delta_x_qp_inf_norm_list_.reserve(static_cast<std::size_t>(max_iter));
        tail_solve_delta_z_qp_inf_norm_list_.reserve(static_cast<std::size_t>(max_iter));
      };

      void Reset()
      {
        Base::SolverInfo::Reset();
        primal_residual_kinematics_list_.clear();
        primal_residual_task_list_.clear();
        primal_residual_slack_list_.clear();

        dual_residual_v_list_.clear();
        dual_residual_nu_list_.clear();

        mu_eq_list_.clear();
        mu_ineq_list_.clear();

        tail_solve_iter_list_.clear();
        tail_solve_primal_residual_list_.clear();
        tail_solve_primal_residual_kinematics_list_.clear();
        tail_solve_primal_residual_task_list_.clear();
        tail_solve_primal_residual_slack_list_.clear();

        tail_solve_dual_residual_list_.clear();
        tail_solve_dual_residual_v_list_.clear();
        tail_solve_dual_residual_nu_list_.clear();

        tail_solve_delta_x_qp_inf_norm_list_.clear();
        tail_solve_delta_z_qp_inf_norm_list_.clear();
      }

      std::vector<Scalar> primal_residual_kinematics_list_;
      std::vector<Scalar> primal_residual_task_list_;
      std::vector<Scalar> primal_residual_slack_list_;

      std::vector<Scalar> dual_residual_v_list_;
      std::vector<Scalar> dual_residual_nu_list_;

      std::vector<Scalar> mu_eq_list_;
      std::vector<Scalar> mu_ineq_list_;

      // for tail solve iterations
      std::vector<int> tail_solve_iter_list_;
      std::vector<Scalar> tail_solve_primal_residual_list_;
      std::vector<Scalar> tail_solve_primal_residual_kinematics_list_;
      std::vector<Scalar> tail_solve_primal_residual_task_list_;
      std::vector<Scalar> tail_solve_primal_residual_slack_list_;

      std::vector<Scalar> tail_solve_dual_residual_list_;
      std::vector<Scalar> tail_solve_dual_residual_v_list_;
      std::vector<Scalar> tail_solve_dual_residual_nu_list_;

      std::vector<Scalar> tail_solve_delta_x_qp_inf_norm_list_;
      std::vector<Scalar> tail_solve_delta_z_qp_inf_norm_list_;
    };

    FirstOrderLoikOptimizedTpl(
      const int max_iter,
      const Scalar & tol_abs,
      const Scalar & tol_rel,
      const Scalar & tol_primal_inf,
      const Scalar & tol_dual_inf,
      const Scalar & rho,
      const Scalar & mu,
      const Scalar & mu_equality_scale_factor,
      const ADMMPenaltyUpdateStrat & mu_update_strat,
      const int num_eq_c,
      const int eq_c_dim,
      const Model & model,
      IkIdData & ik_id_data,
      const bool warm_start,
      const Scalar tol_tail_solve,
      const bool verbose,
      const bool logging)
    : Base(
        max_iter,
        tol_abs,
        tol_rel,
        tol_primal_inf,
        tol_dual_inf,
        rho,
        mu,
        mu_equality_scale_factor,
        mu_update_strat,
        verbose,
        logging)
    , model_(model)
    , ik_id_data_(ik_id_data)
    , problem_(model.njoints, model.njoints - 1, num_eq_c, eq_c_dim, model.nv)
    , nj_(model.njoints)
    , nb_(model.njoints - 1)
    , nv_(model.nv)
    , warm_start_(warm_start)
    , tol_tail_solve_(tol_tail_solve)
    , loik_solver_info_(max_iter)
    {
      // initialize helper quantities
      joint_full_range_ = ik_id_data_.joint_full_range; // [0, nj - 1]
      joint_range_ = ik_id_data_.joint_range;           // [1, nj - 1]

      // residual vectors
      // primal_residual_vec_ = std::numeric_limits<Scalar>::infinity() * DVec::Ones(6 * nb_ + nv_);
      // dual_residual_vec_ = std::numeric_limits<Scalar>::infinity() * DVec::Ones(6 * nb_ + nv_);

      primal_residual_vec_ = DVec::Zero(6 * nb_ + nv_);
      dual_residual_vec_ = DVec::Zero(6 * nb_ + nv_);

      ResetSolver();
    };

    ///
    /// \brief Reset the diff IK solver
    ///
    void ResetSolver()
    {

      Base::Reset(); // reset base

      tail_solve_iter_ = 0;

      delta_x_qp_inf_norm_ = 0.0;
      delta_y_qp_inf_norm_ = 0.0;
      A_qp_T_delta_y_qp_inf_norm_ = 0.0;
      ub_qp_T_delta_y_qp_plus_ = 0.0;
      lb_qp_T_delta_y_qp_minus_ = 0.0;
      primal_infeasibility_cond_1_ = false;
      primal_infeasibility_cond_2_ = false;

      mu_eq_ = this->mu_equality_scale_factor_
               * this->mu_; // 'mu_' is reset to 'mu0_' during 'Base::Reset()'
      mu_ineq_ = this->mu_;

    }; // ResetSolver

    ///
    /// \brief Initial forward pass, to propagate forward kinematics.
    ///
    void FwdPassInit(const DVec & q);

    ///
    /// \brief LOIK first forward pass
    ///
    void FwdPass1();

    ///
    /// \brief LOIK first packward pass optimized as visitor
    ///
    void BwdPassOptimizedVisitor();

    ///
    /// \brief LOIK second forward pass optimized as visitor
    ///
    void FwdPass2OptimizedVisitor();

    ///
    /// \brief Box projection of primal and slack composite quantites
    ///
    void BoxProj();

    ///
    /// \brief ADMM dual variable updates
    ///
    void DualUpdate();

    ///
    /// \brief LOIK second backward pass optimized as visitor
    ///
    void BwdPass2OptimizedVisitor();

    ///
    /// \brief Compute primal residuals
    ///
    void ComputePrimalResiduals();

    ///
    /// \brief Compute dual residuals
    ///
    void ComputeDualResiduals();

    ///
    /// \brief Compute solver residuals
    ///
    void ComputeResiduals();

    ///
    /// \brief Check primal and dual convergence
    ///
    void CheckConvergence();

    ///
    /// \brief Check primal and dual feasibility
    ///
    void CheckFeasibility();

    ///
    /// \brief Update ADMM penalty mu
    ///
    void UpdateMu();

    ///
    /// \brief when infeasibility is detected, run tail solve so primal residual converges to
    /// something.
    ///        This gives theoretical guarantee that the solution (unprojected) converges the
    ///        closest feasible solution.
    ///
    void InfeasibilityTailSolve()
    {
      tail_solve_iter_ = 0;

      while (delta_x_qp_inf_norm_ >= tol_tail_solve_
             || ik_id_data_.delta_z_inf_norm >= tol_tail_solve_)
      {
        if (this->iter_ >= this->max_iter_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: tail solve "
                         "exceed max_iter_: "
                      << tail_solve_iter_ << " iterations." << std::endl;
            std::cerr
              << "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: normInf delta_x_qp_: "
              << delta_x_qp_inf_norm_ << std::endl;
            std::cerr
              << "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: normInf delta_z_qp_: "
              << ik_id_data_.delta_z_inf_norm << std::endl;
          }
          return;
        }

        this->iter_++;
        loik_solver_info_.iter_list_.push_back(this->iter_);

        tail_solve_iter_++;
        loik_solver_info_.tail_solve_iter_list_.push_back(tail_solve_iter_);

        ik_id_data_.UpdatePrev();

        ik_id_data_.ResetInfNorms();

        FwdPass1();

        BwdPassOptimizedVisitor();

        FwdPass2OptimizedVisitor();

        BoxProj();

        DualUpdate();

        ComputeResiduals();

        delta_x_qp_inf_norm_ =
          std::max(ik_id_data_.delta_vis_inf_norm, ik_id_data_.delta_nu_inf_norm);
      }

      if (this->verbose_)
      {
        std::cerr
          << "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: tail solve completed after "
          << tail_solve_iter_ << " iterations." << std::endl;
        std::cerr << "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: normInf delta_x_qp_: "
                  << delta_x_qp_inf_norm_ << std::endl;
        std::cerr << "[FirstOrderLoikOptimizedTpl::InfeasibilityTailSolve]: normInf delta_z_qp_: "
                  << ik_id_data_.delta_z_inf_norm << std::endl;
      }

    }; // InfeasibilityTailSolve

    ///
    /// \brief Initialize the problem to be solved.
    ///
    /// \param[in] q                               current generalized configuration  (DVec)
    /// \param[in] H_ref                           Cost weight for tracking reference (DMat)
    /// \param[in] v_ref                           reference spatial velocity (DVec)
    /// \param[in] active_task_constraint_ids      vector of joint ids where equality constraints
    /// are present (std::vector) \param[in] Ais                             vector of equality
    /// constraint matrix (std::vector) \param[in] bis                             vector of
    /// equality constraint targets (std::vector) \param[in] lb                              joint
    /// velocity lower bounds (DVec) \param[in] ub                              joint velocity upper
    /// bounds (DVec) \param[out] this->ik_id_data_.z            projected joint velocities onto the
    /// box constraint set
    ///
    void SolveInit(
      const DVec & q,
      const Mat6x6 & H_ref,
      const Motion & v_ref,
      const std::vector<Index> & active_task_constraint_ids,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) & Ais,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) & bis,
      const DVec & lb,
      const DVec & ub)
    {
      // reset logging if this->logging_
      if (this->logging_)
      {
        loik_solver_info_.Reset(); // reset logging
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

    }; // SolveInit

    ///
    /// \brief Solve the constrained differential IK problem, just the main loop,
    ///        useful mainly for checking computation timings
    ///
    void Solve()
    {
      // reset IkIdData recursion quantites
      ik_id_data_.ResetRecursion();

      // wipe solver quantities'
      ResetSolver();

      // solver main loop
      for (int i = 1; i < this->max_iter_; i++)
      {

        this->iter_ = i;

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

        if (this->logging_)
        {

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
        if (this->iter_ > 1)
        {
          CheckFeasibility();
        }

        if (this->converged_)
        {
          break; // converged, break out of solver main loop
        }
        else if (this->primal_infeasible_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: primal infeasibility "
                         "detected at iteration: "
                      << this->iter_ << std::endl;
          }
          // problem is primal infeasible, run infeasibility tail solve
          InfeasibilityTailSolve();
          break;
        }
        else if (this->dual_infeasible_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: dual infeasibility detected "
                         "at iteration: "
                      << this->iter_ << std::endl;
          }
          // problem is dual infeasibile, run infeasibility tail solve
          InfeasibilityTailSolve();
          break;
        }

        // update ADMM penalty
        UpdateMu();
      }
    }; // Solve

    ///
    /// \brief Stand alone Solve, solves the constrained differential IK problem.
    ///
    /// Attention, this `Solve()` call will wipe the problem formulation everytime, therefore
    /// not the most efficient implementation, consider using tailored `Solve()` for specific
    /// scenarios such as trakectory tracking.
    ///
    /// \param[in] q                               current generalized configuration  (DVec)
    /// \param[in] H_ref                           Cost weight for tracking reference (DMat)
    /// \param[in] v_ref                           reference spatial velocity (DVec)
    /// \param[in] active_task_constraint_ids      vector of joint ids where equality constraints
    /// are present (std::vector) \param[in] Ais                             vector of equality
    /// constraint matrix (std::vector) \param[in] bis                             vector of
    /// equality constraint targets (std::vector) \param[in] lb                              joint
    /// velocity lower bounds (DVec) \param[in] ub                              joint velocity upper
    /// bounds (DVec) \param[out] this->ik_id_data_.z            projected joint velocities onto the
    /// box constraint set
    ///
    void Solve(
      const DVec & q,
      const Mat6x6 & H_ref,
      const Motion & v_ref,
      const std::vector<Index> & active_task_constraint_ids,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) & Ais,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) & bis,
      const DVec & lb,
      const DVec & ub)
    {
      // reset logging if this->logging_
      if (this->logging_)
      {
        loik_solver_info_.Reset(); // reset logging
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

      // solver main loop
      for (int i = 1; i < this->max_iter_; i++)
      {

        this->iter_ = i;

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

        if (this->logging_)
        {

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
        if (this->iter_ > 1)
        {
          CheckFeasibility();
        }

        if (this->converged_)
        {
          break; // converged, break out of solver main loop
        }
        else if (this->primal_infeasible_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: primal infeasibility "
                         "detected at iteration: "
                      << this->iter_ << std::endl;
          }
          // problem is primal infeasible, run infeasibility tail solve
          InfeasibilityTailSolve();
          break;
        }
        else if (this->dual_infeasible_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: dual infeasibility detected "
                         "at iteration: "
                      << this->iter_ << std::endl;
          }
          // problem is dual infeasibile, run infeasibility tail solve
          InfeasibilityTailSolve();
          break;
        }

        // update ADMM penalty
        UpdateMu();
      }
    }; // Solve general purpose

    ///
    /// \brief Stand alone Solve, solves the constrained differential IK problem.
    ///
    /// Attention, this `Solve()` call will wipe the problem formulation everytime, therefore
    /// not the most efficient implementation, consider using tailored `Solve()` for specific
    /// scenarios such as trakectory tracking.
    ///
    /// \param[in] q                               current generalized configuration  (DVec)
    /// \param[in] c_id                            joint ids where equality constraint need to be
    /// updated \param[in] Ais                             equality constraint matrix (Mat6x6)
    /// \param[in] bis                             equality constraint target (Vec6)
    /// \param[out] this->ik_id_data_.z            projected joint velocities onto the box
    /// constraint set
    ///
    void Solve(const DVec & q, const Index c_id, const Mat6x6 & Ai, const Vec6 & bi)
    {
      // reset logging if this->logging_
      if (this->logging_)
      {
        loik_solver_info_.Reset(); // reset logging
      }

      // reset IkIdData
      ik_id_data_.Reset(this->warm_start_);

      // wipe solver quantities'
      ResetSolver();

      // update problem formulation
      problem_.UpdateEqConstraint(c_id, Ai, bi);

      FwdPassInit(q);

      // solver main loop
      for (int i = 1; i < this->max_iter_; i++)
      {

        this->iter_ = i;

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

        if (this->logging_)
        {

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
        if (this->iter_ > 1)
        {
          CheckFeasibility();
        }

        if (this->converged_)
        {
          break; // converged, break out of solver main loop
        }
        else if (this->primal_infeasible_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: primal infeasibility "
                         "detected at iteration: "
                      << this->iter_ << std::endl;
          }
          // problem is primal infeasible, run infeasibility tail solve
          InfeasibilityTailSolve();
          break;
        }
        else if (this->dual_infeasible_)
        {
          if (this->verbose_)
          {
            std::cerr << "WARNING [FirstOrderLoikOptimizedTpl::Solve]: dual infeasibility detected "
                         "at iteration: "
                      << this->iter_ << std::endl;
          }
          // problem is dual infeasibile, run infeasibility tail solve
          InfeasibilityTailSolve();
          break;
        }

        // update ADMM penalty
        UpdateMu();
      }

    }; // Solver tailored

    inline DVec get_primal_residual_vec() const
    {
      return primal_residual_vec_;
    };
    inline DVec get_dual_residual_vec() const
    {
      return dual_residual_vec_;
    };
    inline Scalar get_dual_residual_v() const
    {
      return dual_residual_v_;
    };
    inline Scalar get_dual_residual_nu() const
    {
      return dual_residual_nu_;
    };
    inline Scalar get_tol_tail_solve() const
    {
      return tol_tail_solve_;
    };
    inline void set_tol_tail_solve(const Scalar tol)
    {
      tol_tail_solve_ = tol;
    };

    /// Debug utility functions
    inline Scalar get_delta_x_qp_inf_norm()
    {
      delta_x_qp_inf_norm_ =
        std::max(ik_id_data_.delta_vis_inf_norm, ik_id_data_.delta_nu_inf_norm);
      return delta_x_qp_inf_norm_;
    };

    inline Scalar get_delta_z_qp_inf_norm()
    {
      return ik_id_data_.delta_z_inf_norm;
    };

    inline Scalar get_delta_y_qp_inf_norm()
    {
      delta_y_qp_inf_norm_ = std::max(
        ik_id_data_.delta_fis_inf_norm,
        std::max(ik_id_data_.delta_yis_inf_norm, ik_id_data_.delta_w_inf_norm));
      return delta_y_qp_inf_norm_;
    };

    inline Scalar get_A_qp_T_delta_y_qp_inf_norm()
    {
      A_qp_T_delta_y_qp_inf_norm_ = std::max(
        ik_id_data_.delta_fis_diff_plus_Aty_inf_norm, ik_id_data_.delta_Stf_plus_w_inf_norm);

      return A_qp_T_delta_y_qp_inf_norm_;
    };

    inline Scalar get_ub_qp_T_delta_y_qp_plus()
    {
      ub_qp_T_delta_y_qp_plus_ = ik_id_data_.bT_delta_y_plus;
      ub_qp_T_delta_y_qp_plus_ += (problem_.ub_.transpose() * ik_id_data_.delta_w.cwiseMax(0))[0];
      return ub_qp_T_delta_y_qp_plus_;
    };

    inline Scalar get_lb_qp_T_delta_y_qp_minus()
    {
      lb_qp_T_delta_y_qp_minus_ = ik_id_data_.bT_delta_y_minus;
      lb_qp_T_delta_y_qp_minus_ += (problem_.lb_.transpose() * ik_id_data_.delta_w.cwiseMin(0))[0];
      return lb_qp_T_delta_y_qp_minus_;
    };

    inline bool get_primal_infeasibility_cond_1()
    {
      return A_qp_T_delta_y_qp_inf_norm_ <= this->tol_primal_inf_ * delta_y_qp_inf_norm_;
    };

    inline bool get_primal_infeasibility_cond_2()
    {
      return (ub_qp_T_delta_y_qp_plus_ + lb_qp_T_delta_y_qp_minus_)
             <= this->tol_primal_inf_ * delta_y_qp_inf_norm_;
    };

  protected:
    Model model_;
    IkIdData & ik_id_data_;

    ProblemFormulation problem_;

    // ADMM solver specific quantities
    int tail_solve_iter_;               // tail solve iteration index
    Scalar primal_residual_kinematics_; // primal residual of just the forward kinematics equality
                                        // constraints
    Scalar primal_residual_task_;       // primal residual of just the task equality constraints
    Scalar primal_residual_slack_; // primal residual of just the inequality induced slack equality
                                   // constraints
    DVec primal_residual_vec_;     // utility vector for primal residual calculation

    Scalar dual_residual_v_;  // dual residual of just the dual feasibility condition wrt v
    Scalar dual_residual_nu_; // dual residual of just the dual feasibility condition wrt nu
    DVec dual_residual_vec_;  // utility vector for dual residual calculation

    // test utilities
    Scalar delta_x_qp_inf_norm_;
    Scalar delta_y_qp_inf_norm_;
    Scalar A_qp_T_delta_y_qp_inf_norm_;
    Scalar ub_qp_T_delta_y_qp_plus_;
    Scalar lb_qp_T_delta_y_qp_minus_;
    bool primal_infeasibility_cond_1_;
    bool primal_infeasibility_cond_2_;

    Scalar mu_eq_;   // ADMM penalty for equality constraints
    Scalar mu_ineq_; // ADMM penalty for inequality constraints

    // solver helper quantities
    int nj_;                    // number of joints in the model_
    int nb_;                    // number of bodies in the model_, 'nb_ = nj_ - 1'
    int nv_;                    // dimension of nu_ (q_dot)
    IndexVector joint_full_range_; // index of full joint range, [0, njoints - 1]
    IndexVector joint_range_; // index of joint range excluding the world/universe [1, njoints - 1]

    // warm_start flag
    bool warm_start_;

    // tol for infeasibility tail solve
    Scalar tol_tail_solve_;

    // solver info logging struct
    LoikSolverInfo loik_solver_info_;
  };

} // namespace loik

#include "loik/loik-loid-optimized.hxx"

#if LOIK_ENABLE_TEMPLATE_INSTANTIATION
  #include "loik/loik-loid-optimized.txx"
#endif // LOIK_ENABLE_TEMPLATE_INSTANTIATION
