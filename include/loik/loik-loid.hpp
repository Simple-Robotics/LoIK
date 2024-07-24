//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/macros.hpp"
#include "loik/ik-id-description.hpp"
#include "loik/loik-loid-data.hpp"
#include "loik/task-solver-base.hpp"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/multibody/fwd.hpp>

namespace loik
{

  template<typename _Scalar>
  struct FirstOrderLoikTpl : IkIdSolverBaseTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef IkIdSolverBaseTpl<_Scalar> Base;
    typedef typename Base::Scalar Scalar;
    using Model = pinocchio::ModelTpl<_Scalar>;
    using IkIdData = IkIdDataTpl<_Scalar>;
    using JointModel = typename IkIdData::JointModel;
    using JointData = typename IkIdData::JointData;
    using ProblemFormulation = IkProblemStandardQPFormulation<_Scalar>;
    using Motion = typename IkIdData::Motion;
    using SE3 = typename IkIdData::SE3;
    using DMat = typename IkIdData::DMat;
    using DVec = typename IkIdData::DVec;
    using Vec3 = typename IkIdData::Vec3;
    using Vec6 = typename IkIdData::Vec6;
    using Mat6x6 = typename IkIdData::Mat6x6;
    using Index = typename IkIdData::Index;
    using IndexVec = typename IkIdData::IndexVector;

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

      // TODO: maybe record the dual variables
    };

    FirstOrderLoikTpl(
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

      ResetSolver();
    };

    ///
    /// \brief Reset the diff IK solver
    ///
    void ResetSolver()
    {
      loik_solver_info_.Reset(); // reset logging
      problem_.Reset();          // reset problem formulation
      Base::Reset();             // reset base

      ik_id_data_.Reset(this->warm_start_); // reset IkIdData

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

      primal_residual_vec_ = 0.0 * DVec::Ones(problem_.eq_c_dim_ * nb_ + nv_);
      dual_residual_vec_ = 0.0 * DVec::Ones(6 * nb_ + nv_);

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
    /// \brief LOIK first packward pass
    ///
    void BwdPass();

    ///
    /// \brief LOIK second forward pass
    ///
    void FwdPass2();

    ///
    /// \brief Box projection of primal and slack composite quantites
    ///
    void BoxProj();

    ///
    /// \brief ADMM dual variable updates
    ///
    void DualUpdate();

    ///
    /// \brief unit test utility function
    ///
    void UpdateQPADMMSolveLoopUtility();

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

      while (problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() >= tol_tail_solve_
             || problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() >= tol_tail_solve_)
      {

        if (this->iter_ >= this->max_iter_)
        {
          if (this->verbose_)
          {
            std::cerr
              << "WARNING [FirstOrderLoik::InfeasibilityTailSolve]: tail solve exceeds max_iter_: "
              << tail_solve_iter_ << " iterations." << std::endl;
            std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: normInf delta_x_qp_: "
                      << problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
            std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: normInf delta_z_qp_: "
                      << problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
          }
          return;
        }

        this->iter_++;

        if (this->verbose_)
        {
          std::cout << "===============" << std::endl;
          std::cout << "ADMM iter: " << this->iter_ << "||" << std::endl;
          std::cout << "===============" << std::endl;
        }

        loik_solver_info_.iter_list_.push_back(this->iter_);

        tail_solve_iter_++;
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

        if (this->logging_)
        {

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
          loik_solver_info_.tail_solve_primal_residual_slack_list_.push_back(
            primal_residual_slack_);
          loik_solver_info_.tail_solve_primal_residual_list_.push_back(this->primal_residual_);
          loik_solver_info_.tail_solve_dual_residual_nu_list_.push_back(dual_residual_nu_);
          loik_solver_info_.tail_solve_dual_residual_v_list_.push_back(dual_residual_v_);
          loik_solver_info_.tail_solve_dual_residual_list_.push_back(this->dual_residual_);
          loik_solver_info_.tail_solve_delta_x_qp_inf_norm_list_.push_back(
            problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>());
          loik_solver_info_.tail_solve_delta_z_qp_inf_norm_list_.push_back(
            problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>());
        }
      }

      if (this->verbose_)
      {
        std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: tail solve completed after "
                  << tail_solve_iter_ << " iterations." << std::endl;
        std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: normInf delta_x_qp_: "
                  << problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
        std::cerr << "[FirstOrderLoik::InfeasibilityTailSolve]: normInf delta_z_qp_: "
                  << problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
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
      // wipe everything, warm-start is taken care of by 'ik_id_data_.Reset()'
      ResetSolver();

      // TODO: perform initial fwd pass, to calculate forward kinematics quantities
      FwdPassInit(q);

      // update problem formulation
      problem_.UpdateQPADMMSolveInit(
        H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub, model_, ik_id_data_);

    }; // SolveInit

    ///
    /// \brief Solve the constrained differential IK problem, just the main loop
    ///
    void Solve()
    {
      // solver main loop
      for (int i = 1; i < this->max_iter_; i++)
      {

        this->iter_ = i;

        loik_solver_info_.iter_list_.push_back(this->iter_);

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

        if (this->logging_)
        {

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
            std::cerr
              << "WARNING [FirstOrderLoik::Solve]: primal infeasibility detected at iteration: "
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
            std::cerr
              << "WARNING [FirstOrderLoik::Solve]: dual infeasibility detected at iteration: "
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
      // wipe everything, warm-start is taken care of by 'ik_id_data_.Reset()'
      ResetSolver();

      // TODO: perform initial fwd pass, to calculate forward kinematics quantities
      FwdPassInit(q);

      // update problem formulation
      problem_.UpdateQPADMMSolveInit(
        H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub, model_, ik_id_data_);

      // solver main loop
      for (int i = 1; i < this->max_iter_; i++)
      {

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

        // box projection
        BoxProj();

        // dual update
        DualUpdate();

        // update standard qp formulation using primal dual variables from current iter
        problem_.UpdateQPADMMSolveLoop(ik_id_data_);

        // compute residuals
        ComputeResiduals();

        if (this->logging_)
        {

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
            std::cerr
              << "WARNING [FirstOrderLoik::Solve]: primal infeasibility detected at iteration: "
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
            std::cerr
              << "WARNING [FirstOrderLoik::Solve]: dual infeasibility detected at iteration: "
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

    /// test utilities
    inline Scalar get_delta_x_qp_inf_norm() const
    {
      return problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>();
    };
    inline Scalar get_delta_z_qp_inf_norm() const
    {
      return problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>();
    };
    inline DVec get_delta_y_qp() const
    {
      return problem_.delta_y_qp_;
    };
    inline Scalar get_delta_y_qp_inf_norm() const
    {
      return problem_.delta_y_qp_.template lpNorm<Eigen::Infinity>();
    };
    inline DVec get_A_qp_delta_y_qp() const
    {
      return problem_.A_qp_ * problem_.delta_y_qp_;
    };
    inline Scalar get_A_qp_T_delta_y_qp_inf_norm() const
    {
      return (problem_.A_qp_.transpose() * problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();
    };
    inline Scalar get_ub_qp_T_delta_y_qp_plus() const
    {
      return (problem_.ub_qp_.transpose() * problem_.delta_y_qp_plus_).value();
    };
    inline Scalar get_lb_qp_T_delta_y_qp_minus() const
    {
      return (problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_).value();
    };

    bool get_primal_infeasibility_cond_1() const
    {
      return (problem_.A_qp_.transpose() * problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>()
             <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();
    };

    bool get_primal_infeasibility_cond_2() const
    {
      return (problem_.ub_qp_.transpose() * problem_.delta_y_qp_plus_
              + problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_)
               .value()
             <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();
    };

  protected:
    const Model & model_;
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

    Scalar dual_residual_prev_;
    Scalar delta_dual_residual_;
    Scalar dual_residual_v_; // dual residual of just the dual feasibility condition wrt v
    Scalar dual_residual_v_prev_;
    Scalar delta_dual_residual_v_;
    Scalar dual_residual_nu_; // dual residual of just the dual feasibility condition wrt nu
    Scalar dual_residual_nu_prev_;
    Scalar delta_dual_residual_nu_;
    DVec dual_residual_vec_; // utility vector for dual residual calculation

    Scalar mu_eq_;   // ADMM penalty for equality constraints
    Scalar mu_ineq_; // ADMM penalty for inequality constraints

    // solver helper quantities
    int nj_;                    // number of joints in the model_
    int nb_;                    // number of bodies in the model_, 'nb_ = nj_ - 1'
    int nv_;                    // dimension of nu_ (q_dot)
    IndexVec joint_full_range_; // index of full joint range, [0, njoints - 1]
    IndexVec joint_range_; // index of joint range excluding the world/universe [1, njoints - 1]

    // warm_start flag
    bool warm_start_;

    // tol for infeasibility tail solve
    Scalar tol_tail_solve_;

    // solver info logging struct
    LoikSolverInfo loik_solver_info_;
  };

} // namespace loik

#include "loik/loik-loid.hxx"

#if LOIK_ENABLE_TEMPLATE_INSTANTIATION
  #include "loik/loik-loid.txx"
#endif // LOIK_ENABLE_TEMPLATE_INSTANTIATION
