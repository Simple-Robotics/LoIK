//
// Copyright (c) 2024 INRIA
//

#pragma once

#include <pinocchio/math/comparison-operators.hpp>
#include <pinocchio/math/fwd.hpp>

namespace loik
{

  enum ADMMPenaltyUpdateStrat
  {
    DEFAULT = 0,
    OSQP = 1,
    MAXEIGENVALUE = 3
  }; // enum ADMMPenaltyUpdateStrat

  template<typename _Scalar>
  struct IkIdSolverBaseTpl
  {
    typedef _Scalar Scalar;

    struct SolverInfo
    {
      explicit SolverInfo(const int max_iter)
      {
        primal_residual_list_.reserve(static_cast<std::size_t>(max_iter));
        dual_residual_list_.reserve(static_cast<std::size_t>(max_iter));
        mu_list_.reserve(static_cast<std::size_t>(max_iter));
        iter_list_.reserve(static_cast<std::size_t>(max_iter));
      };

      void Reset()
      {
        primal_residual_list_.clear();
        dual_residual_list_.clear();
        mu_list_.clear();
        iter_list_.clear();
      };

      int Size() const
      {
        return iter_list_.size();
      }

      std::vector<Scalar> primal_residual_list_;
      std::vector<Scalar> dual_residual_list_;
      std::vector<Scalar> mu_list_;
      std::vector<int> iter_list_;
    };

    IkIdSolverBaseTpl(
      const int max_iter,
      const Scalar & tol_abs,
      const Scalar & tol_rel,
      const Scalar & tol_primal_inf,
      const Scalar & tol_dual_inf,
      const Scalar & rho,
      const Scalar & mu,
      const Scalar & mu_equality_scale_factor,
      const ADMMPenaltyUpdateStrat & mu_update_strat,
      const bool verbose,
      const bool logging)
    : rho_(rho)
    , mu0_(mu)
    , mu_(mu)
    , mu_equality_scale_factor_(mu_equality_scale_factor)
    , mu_update_strat_(mu_update_strat)
    , max_iter_(max_iter)
    , tol_abs_(tol_abs)
    , tol_rel_(tol_rel)
    , tol_primal_inf_(tol_primal_inf)
    , tol_dual_inf_(tol_dual_inf)
    , verbose_(verbose)
    , logging_(logging)
    , solver_info_(max_iter) {};

    /// \brief reset solver
    void Reset()
    {

      iter_ = 0;
      converged_ = false;
      primal_infeasible_ = false;
      dual_infeasible_ = false;

      mu_ = mu0_;
    };

    /// \brief get current iteration count
    inline int get_iter() const
    {
      return iter_;
    };

    /// \brief get primal residual at the last finished iteration
    inline Scalar get_primal_residual() const
    {
      return primal_residual_;
    };

    /// \brief get dual residual at the last finished iteration
    inline Scalar get_dual_residual() const
    {
      return dual_residual_;
    };

    /// \brief get convergence status at the last finished iteration
    inline bool get_convergence_status() const
    {
      return converged_;
    };

    /// \brief get primal infeasibility status at the last finished iteration
    inline bool get_primal_infeasibility_status() const
    {
      return primal_infeasible_;
    };

    /// \brief get dual infeasibility status at the last finished iteration
    inline bool get_dual_infeasibility_status() const
    {
      return dual_infeasible_;
    };

    /// \brief set the number of maximum iterations
    inline void set_max_iter(const int max_iter)
    {
      this->max_iter_ = max_iter;
    };

    /// \brief set proximal parameter
    inline void set_rho(const Scalar rho)
    {
      this->rho_ = rho;
    };

    /// \brief get proximal parameter
    inline Scalar get_rho() const
    {
      return rho_;
    };

    /// \brief set ADMM master penalty
    inline void set_mu(const Scalar mu)
    {
      this->mu_ = mu;
    };

    /// \brief get ADMM master penalty
    inline Scalar get_mu() const
    {
      return mu_;
    };

    /// \brief set primal convergence tolerance directly
    inline void set_tol_primal(const Scalar tol_primal)
    {
      this->tol_primal_ = tol_primal;
    };

    /// \brief get primal convergence tolerance
    inline Scalar get_tol_primal() const
    {
      return tol_primal_;
    };

    /// \brief set dual convergence tolerance directly
    inline void set_tol_dual(const Scalar tol_dual)
    {
      this->tol_dual_ = tol_dual;
    };

    /// \brief get dual convergence tolerance
    inline Scalar get_tol_dual() const
    {
      return tol_dual_;
    };

    /// \brief set primal infeasibility tolerance directly
    inline void set_tol_primal_inf(const Scalar tol_primal_inf)
    {
      this->tol_primal_inf_ = tol_primal_inf;
    };

    /// \brief get primal infeasibility tolerance
    inline Scalar get_tol_primal_inf() const
    {
      return tol_primal_inf_;
    };

    /// \brief set dual infeasibility tolerance directly
    inline void set_tol_dual_inf(const Scalar tol_dual_inf)
    {
      this->tol_dual_inf_ = tol_dual_inf;
    };

    /// \brief get dual infeasibility tolerance
    inline Scalar get_tol_dual_inf() const
    {
      return tol_dual_inf_;
    };

  protected:
    Scalar rho_; // proximal parameter for primal vars solved in the equality constraint QP step
    Scalar mu0_; // initial ADMM master penalty parameter
    Scalar mu_;  // ADMM master penalty parameter
    Scalar mu_equality_scale_factor_; // ADMM penalty scaling factor for equality constraints

    ADMMPenaltyUpdateStrat mu_update_strat_; // enum to switch between mu update strategies

    int max_iter_;           // solver maximum iteration
    int iter_;               // current iteration
    bool converged_;         // convergence flag
    Scalar tol_abs_;         // absolute tol
    Scalar tol_rel_;         // relative tol
    Scalar tol_primal_;      // primal residual convergence tol
    Scalar tol_dual_;        // dual residual convergence tol
    Scalar tol_primal_inf_;  // primal infeasibility tol
    Scalar tol_dual_inf_;    // dual infeasibility tol
    bool primal_infeasible_; // primal infeasibility flag
    bool dual_infeasible_;   // dual infeasibility flag

    Scalar primal_residual_; // primal residual at current iter
    Scalar dual_residual_;   // dual residual at current iter

    bool verbose_;           // verbose printing
    bool logging_;           // solver data logging
    SolverInfo solver_info_; // solver logging info

  }; // struct IkIdSolverBaseTpl

} // namespace loik
