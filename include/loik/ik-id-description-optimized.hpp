//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "fwd.hpp"
#include "loik/macros.hpp"
#include "loik/loik-loid-data-optimized.hpp"

#include <pinocchio/math/comparison-operators.hpp>
#include <pinocchio/math/fwd.hpp>

namespace loik
{

  template<typename _Scalar>
  struct IkProblemFormulationOptimized
  {
    using IkIdDataOptimized = IkIdDataTypeOptimizedTpl<_Scalar>;
    IKID_DATA_TYPEDEF_TEMPLATE(IkIdDataOptimized);
    

    explicit IkProblemFormulationOptimized(
      const int nj, const int nb, const int nc_eq, const int eq_c_dim, const int ineq_c_dim)
    : eq_c_dim_(eq_c_dim)
    , nj_(nj)
    , nb_(nb)
    , nc_eq_(nc_eq)
    , ineq_c_dim_(ineq_c_dim)
    {
      if (nj_ != nb_ + 1)
      {
        throw(std::runtime_error("[IkProblemFormulation::IkProblemFormulation]: nb does not equal "
                                 "to nj - 1, robot model not supported !!!"));
      }

      if (eq_c_dim != 6)
      {
        throw(
          std::runtime_error("[IkProblemFormulation::IkProblemFormulation]: equality constraint "
                             "dimension is not 6, problem formulation not supported !!!"));
      }

      H_refs_.reserve(static_cast<std::size_t>(nj));
      v_refs_.reserve(static_cast<std::size_t>(nj));

      active_task_constraint_ids_.reserve(static_cast<std::size_t>(nj));
      Ais_.reserve(static_cast<std::size_t>(nj));
      bis_.reserve(static_cast<std::size_t>(nj));

      lb_ = Eigen::VectorXd::Zero(ineq_c_dim_);
      ub_ = Eigen::VectorXd::Zero(ineq_c_dim_);

      Reset();
    };

    void Reset()
    {

      // reset reference
      ResetReferences();

      // reset equality constraints
      ResetEqConstraints();

      // reset inequality constraints
      ResetIneqConstraints();
    };

    ///
    /// \brief set tracking reference by duplicating weights and targets for all body links
    ///
    void UpdateReference(const Mat6x6 & H_ref_target, const Motion & v_ref_target)
    {
      // check H_refs_ is not empty
      if (H_refs_.size() <= 0)
      {
        throw(std::runtime_error(
          "[IkProblemFormulation::UpdateReference]: 'H_refs_' and 'v_refs_' are empty"));
      }

      for (Index idx = 0; idx < H_refs_.size(); idx++)
      {
        H_refs_[idx] = H_ref_target;
        v_refs_[idx] = v_ref_target;

        LOIK_EIGEN_MALLOC_NOT_ALLOWED();
        Hv[idx] = H_refs_[idx] * v_refs_[idx].toVector();
        LOIK_EIGEN_MALLOC_ALLOWED();
      }

      LOIK_EIGEN_MALLOC_NOT_ALLOWED();
      Hv_inf_norm_ = Hv[0].template lpNorm<Eigen::Infinity>();
      LOIK_EIGEN_MALLOC_ALLOWED();
    };

    ///
    /// \brief set tracking references uisng vectors of references
    ///
    void UpdateReferences(
      const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) & H_refs,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Motion) & v_refs)
    {
      if (
        H_refs.size() != static_cast<std::size_t>(nj_)
        || v_refs.size() != static_cast<std::size_t>(nj_))
      {
        throw(std::runtime_error("[IkProblemFormulation::UpdateReferences]: input arguments "
                                 "'H_refs', 'v_refs' have wrong size!!"));
      }
      H_refs_ = H_refs;
      v_refs_ = v_refs;

      for (Index idx = 0; idx < H_refs_.size(); idx++)
      {
        LOIK_EIGEN_MALLOC_NOT_ALLOWED();
        Hv[idx] = H_refs_[idx] * v_refs_[idx].toVector();

        if (Hv[idx].template lpNorm<Eigen::Infinity>() > Hv_inf_norm_)
        {
          Hv_inf_norm_ = Hv[idx].template lpNorm<Eigen::Infinity>();
        }

        LOIK_EIGEN_MALLOC_ALLOWED();
      }
    };

    ///
    /// \brief batch update equality constraints, number of equality constraints and constraint
    /// dimensions must not change
    ///
    void UpdateEqConstraints(
      const std::vector<Index> & active_task_constraint_ids,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) & Ais,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) & bis)
    {
      // check constraint specification is consistant
      if (!((active_task_constraint_ids.size() == Ais.size()) && ((Ais.size() == bis.size()))))
      {
        throw(std::runtime_error("[IkProblemFormulation::UpdateEqConstraints]: "
                                 "task_constraint_ids, Ais, and bis have different size !!!"));
      }

      // check constraint dimension is consistent between input arguments!
      if (Ais[0].rows() != bis[0].rows())
      {
        throw(std::runtime_error(
          "[IkProblemFormulation::UpdateEqConstraints] Ai and bi dimension mismatch!!"));
      }

      // check if number of equality constraints has changed
      if (active_task_constraint_ids.size() != static_cast<std::size_t>(nc_eq_))
      {
        throw(std::runtime_error("[IkProblemFormulation::UpdateEqConstraints]: number of equality "
                                 "constraints doesn't match initialization!!!"));
      }

      // check if equality constraint dim has changed
      if (Ais[0].rows() != eq_c_dim_)
      {
        // TODO: need to check constraint dimension for each Ai and bi, not just the first ones
        throw(std::runtime_error("[IkProblemFormulation::UpdateEqConstraints]: equality constraint "
                                 "dimension has changed!!! Updating constraint dimension"));
      }

      active_task_constraint_ids_ = active_task_constraint_ids;
      Ais_ = Ais;
      bis_ = bis;

      // set bi_inf_norm to zero just to be sure
      bis_inf_norm_ = 0.0;

      for (Index idx = 0; idx < Ais_.size(); idx++)
      {
        LOIK_EIGEN_MALLOC_NOT_ALLOWED();
        AtA[idx] = Ais_[idx].transpose() * Ais_[idx];
        Atb[idx] = Ais_[idx].transpose() * bis_[idx];

        if (bis_[idx].template lpNorm<Eigen::Infinity>() > bis_inf_norm_)
        {
          bis_inf_norm_ = bis_[idx].template lpNorm<Eigen::Infinity>();
        }

        LOIK_EIGEN_MALLOC_ALLOWED();
      }
    };

    ///
    /// \brief set equality constraints by individual link id and updating 'Ai' and 'bi', the link
    /// id to update must already be
    ///        present in 'active_task_constraint_ids_'.
    ///
    void UpdateEqConstraint(const Index c_id, const Mat6x6 & Ai, const Vec6 & bi)
    {
      // check if constraint already exist at link 'c_id'
      auto found_it =
        std::find(active_task_constraint_ids_.begin(), active_task_constraint_ids_.end(), c_id);

      // if 'c_id' not present in 'active_task_constraint_ids_', invoke 'AddEqConstraint()'
      if (found_it == active_task_constraint_ids_.end())
      {
        throw(std::runtime_error("[IkProblemFormulation::UpdateEqConstraint]: constraint doesn't "
                                 "yet exist at link 'c_id' !!! "));
      }

      // check if constraint dimension is consistent
      if (Ai.rows() != bi.size() || Ai.rows() != static_cast<std::size_t>(eq_c_dim_))
      {
        throw(std::runtime_error(
          "[IkProblemFormulation::UpdateEqConstraint]: constraint dimension inconsistent!!!"));
      }

      // count how many times 'c_id' appear in 'active_task_constraint_ids_'
      int c_id_count =
        std::count(active_task_constraint_ids_.begin(), active_task_constraint_ids_.end(), c_id);

      // if 'c_id' appear more than once, then something went wrong, throw.
      if (c_id_count > 1)
      {
        throw(
          std::runtime_error("[IkProblemFormulation::UpdateEqConstraint]: multiple constraint "
                             "specification for the same link id, not supported, terminating !!!"));
      }

      // get index of 'c_id' in 'active_task_constraint_ids_'
      Index c_id_in_vec = std::distance(active_task_constraint_ids_.begin(), found_it);

      // update Ai
      Ais_[c_id_in_vec] = Ai;
      // update bi
      bis_[c_id_in_vec] = bi;
      // udpate utility
      LOIK_EIGEN_MALLOC_NOT_ALLOWED();
      AtA[c_id_in_vec] = Ai.transpose() * Ai;
      Atb[c_id_in_vec] = Ai.transpose() * bi;

      if (bi.template lpNorm<Eigen::Infinity>() > bis_inf_norm_)
      {
        bis_inf_norm_ = bi.template lpNorm<Eigen::Infinity>();
      }
      LOIK_EIGEN_MALLOC_ALLOWED();
    };

    ///
    /// \brief update equality constriant by link id, update 'bi' only, keep using old 'Ai',
    /// constraint must already be present for link id
    ///
    void UpdateEqConstraint(const Index c_id, const Vec6 & bi)
    {
      // check if constraint already exist at link 'c_id'
      auto found_it =
        std::find(active_task_constraint_ids_.begin(), active_task_constraint_ids_.end(), c_id);

      // if 'c_id' not present in 'active_task_constraint_ids_', then throw
      if (found_it == active_task_constraint_ids_.end())
      {
        throw(std::runtime_error("[IkProblemFormulation::UpdateEqConstraint]: constraint doesn't "
                                 "yet exist at link 'c_id' !!! "));
      }

      // get index of 'c_id' in 'active_task_constraint_ids_'
      Index c_id_in_vec = std::distance(active_task_constraint_ids_.begin(), found_it);
      UpdateEqConstraint(c_id, Ais_[c_id_in_vec], bi);
    }

    ///
    /// \brief (deactivated for now) add equality constraint by link id. If constraint is already
    /// present for link id, then invoke 'UpdateEqConstraint()'
    ///
    void AddEqConstraint(const Index c_id, const Mat6x6 & Ai, const Vec6 & bi)
    {
      // check if 'c_id' in 'active_task_constraint_ids_'
      auto found_it =
        std::find(active_task_constraint_ids_.begin(), active_task_constraint_ids_.end(), c_id);

      // if found, update instead
      if (found_it != active_task_constraint_ids_.end())
      {
        // constraint already defined at link id, updating instead
        UpdateEqConstraint(c_id, Ai, bi);
      }
      else
      {
        // not found

        // check constraint dimension
        if (Ai.rows() != bi.rows())
        {
          throw(std::runtime_error("[IkProblemFormulation::AddEqConstraint]: input arguments "
                                   "constraint dimension inconsistent !!!"));
        }

        if (Ai.rows() != static_cast<std::size_t>(eq_c_dim_))
        {
          throw(std::runtime_error("[IkProblemFormulation::AddEqConstraint]: input constraint "
                                   "dimension differ from existing constriant dimension!!!"));
        }

        active_task_constraint_ids_.push_back(c_id);
        nc_eq_++;

        // add Ai and bi
        Ais_.push_back(Ai);
        bis_.push_back(bi);

        // add to utility
        // LOIK_EIGEN_MALLOC_NOT_ALLOWED();
        AtA.push_back(Ai.transpose() * Ai);
        Atb.push_back(Ai.transpose() * bi);
        // LOIK_EIGEN_MALLOC_ALLOWED();

        // check bis_inf_norm
        if (bi.template lpNorm<Eigen::Infinity>() > bis_inf_norm_)
        {
          bis_inf_norm_ = bi.template lpNorm<Eigen::Infinity>();
        }
      }
    };

    ///
    /// \brief (deactivated for now) remove equality constraint by link id, if constraint not
    /// already present for link id, then do nothing
    ///
    void RemoveEqConstraint(const Index c_id)
    {
      // check if constraint already exist at link 'c_id'
      auto found_it =
        std::find(active_task_constraint_ids_.begin(), active_task_constraint_ids_.end(), c_id);

      if (found_it == active_task_constraint_ids_.end())
      {
        // no constraint defined at input link id, do nothing
        std::cerr << "WARNING [IkProblemFormulation::RemoveEqConstraint]: no constraint defined at "
                     "link id, nothing to remove. "
                  << std::endl;
        return;
      }

      // get index of 'c_id' in 'active_task_constraint_ids_'
      active_task_constraint_ids_.erase(found_it);
      Ais_.erase(found_it);
      bis_.erase(found_it);
      AtA.erase(found_it);
      Atb.erase(found_it);

      bis_inf_norm_ = 0.0;
      for (Index idx = 0; idx < Ais_.size(); idx++)
      {
        if (bis_[idx].template lpNorm<Eigen::Infinity>() > bis_inf_norm_)
        {
          bis_inf_norm_ = bis_[idx].template lpNorm<Eigen::Infinity>();
        }
      }
      // Aty.erase(found_it);
      nc_eq_--;
    };

    ///
    /// \brief set inequality constraints
    ///
    void UpdateIneqConstraints(const DVec & lb, const DVec & ub)
    {
      // check arguments have some dimension
      if (lb.size() != ub.size())
      {
        throw(std::runtime_error("[IkProblemFormulation::UpdateIneqConstraints]: lower bound and "
                                 "upper bound have different dimensions!!!"));
      }

      // check is inequality constraint dimension has changed
      if (lb.size() != ineq_c_dim_)
      {
        throw(
          std::runtime_error("IkProblemFormulation::UpdateIneqConstraints]: inequality constraint "
                             "dimension has changed, this is not supported currently!!!"));
      }

      lb_ = lb;
      ub_ = ub;
    };

    int eq_c_dim_;   // constraint dimension of the equality task constraint, this is shared across
                     // all eq constraints
    int nj_;         // number of joints
    int nb_;         // number of bodies
    int nc_eq_;      // number of equality constraints in the problem
    int ineq_c_dim_; // inequality constraint dimension, == pinocchio Model::nv
    PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)
    H_refs_; // weights for spatial velocity reference tracking in the cost, i.e. soft constraints
    PINOCCHIO_ALIGNED_STD_VECTOR(Motion) v_refs_; // spatial velocity tracking references
    std::vector<Index>
      active_task_constraint_ids_; // vector of idx of links with active hard constraints

    PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)
    Ais_; // 'A' terms in the active equality constraint definitions
    PINOCCHIO_ALIGNED_STD_VECTOR(Vec6)
    bis_;     // 'b' terms in the active equality constraint definitions
    DVec lb_; // lower bound for primal decision variables
    DVec ub_; // upper bound for primal decision variables

    // utility members
    PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) AtA; // Ai.transpose * Ai  ; updated by UpdateEqConstraint
    PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) Atb;   // Ai.transpose * bi  ; updated by UpdateEqConstraint
    // PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) Aty;         // Ai.transpose * yi  ; updated in the solver
    // after dual variables (yis) udpate
    PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) Hv; // Hi_ref_ * vi_ref   ; updated by UpdateReferences
    _Scalar bis_inf_norm_;                 // infinity norm of bis viewd as a vector
    _Scalar Hv_inf_norm_;                  // inf norm of Href * vref viewd as a vector

  private:
    ///
    /// \brief reset tracking references weights and targets
    ///
    void ResetReferences()
    {
      H_refs_.clear();
      v_refs_.clear();
      Hv.clear();
      H_refs_.resize(static_cast<std::size_t>(nj_));
      v_refs_.resize(static_cast<std::size_t>(nj_));
      Hv.resize(static_cast<std::size_t>(nj_));
      for (Index idx = 0; idx < static_cast<std::size_t>(nj_); idx++)
      {
        H_refs_[idx].setZero();
        v_refs_[idx].setZero();
        Hv[idx].setZero();
      }
      Hv_inf_norm_ = 0.0;
    }

    ///
    /// \brief reset equality constraint quantities, 'Ais_' and 'bis_'
    ///
    void ResetEqConstraints()
    {
      active_task_constraint_ids_.clear();
      Ais_.clear();
      bis_.clear();
      AtA.clear();
      Atb.clear();
      bis_inf_norm_ = 0.0;
      // Aty.clear();

      active_task_constraint_ids_.resize(static_cast<std::size_t>(nc_eq_));
      Ais_.resize(static_cast<std::size_t>(nc_eq_));
      bis_.resize(static_cast<std::size_t>(nc_eq_));
      AtA.resize(static_cast<std::size_t>(nc_eq_));
      Atb.resize(static_cast<std::size_t>(nc_eq_));

      for (Index idx = 0; idx < static_cast<std::size_t>(nc_eq_); idx++)
      {
        active_task_constraint_ids_[idx] = 0;
        Ais_[idx].setZero();
        bis_[idx].setZero();
        AtA[idx].setZero();
        Atb[idx].setZero();
      }
    }

    ///
    /// \brief reset inequality constriant quantities, 'lb_' and 'ub_'
    ///
    void ResetIneqConstraints()
    {
      lb_.setZero();
      ub_.setZero();
    }

  }; // struct IkProblemFormulationOptimized

} // namespace loik
