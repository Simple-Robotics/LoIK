//
// Copyright (c) 2024 INRIA
//

#pragma once

#include "loik/loik-loid-data.hpp"

#include <pinocchio/math/comparison-operators.hpp>
#include <pinocchio/math/fwd.hpp>

namespace loik
{

  template<typename _Scalar>
  struct IkProblemFormulation
  {
    using IkIdData = IkIdDataTpl<_Scalar>;
    using Motion = typename IkIdData::Motion;
    using Model = typename IkIdData::Model;
    using DMat = typename IkIdData::DMat;
    using DVec = typename IkIdData::DVec;
    using Vec3 = typename IkIdData::Vec3;
    using Vec6 = typename IkIdData::Vec6;
    using Mat6x6 = typename IkIdData::Mat6x6;
    using Index = typename IkIdData::Index;
    using IndexVec = typename IkIdData::IndexVector;

    explicit IkProblemFormulation(
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
      }
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

  private:
    ///
    /// \brief reset tracking references weights and targets
    ///
    void ResetReferences()
    {
      H_refs_.clear();
      v_refs_.clear();
      H_refs_.resize(static_cast<std::size_t>(nj_));
      v_refs_.resize(static_cast<std::size_t>(nj_));
      for (Index idx = 0; idx < static_cast<std::size_t>(nj_); idx++)
      {
        H_refs_[idx] = Mat6x6::Zero();
        v_refs_[idx] = Motion::Zero();
      }
    }

    ///
    /// \brief reset equality constraint quantities, 'Ais_' and 'bis_'
    ///
    void ResetEqConstraints()
    {
      active_task_constraint_ids_.clear();
      Ais_.clear();
      bis_.clear();

      active_task_constraint_ids_.resize(static_cast<std::size_t>(nc_eq_));
      Ais_.resize(static_cast<std::size_t>(nc_eq_));
      bis_.resize(static_cast<std::size_t>(nc_eq_));

      for (Index idx = 0; idx < static_cast<std::size_t>(nc_eq_); idx++)
      {
        active_task_constraint_ids_[idx] = 0;
        Ais_[idx] = Mat6x6::Zero();
        bis_[idx] = Vec6::Zero();
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

  }; // struct IkProblemFormulation

  template<typename _Scalar>
  struct IkProblemStandardQPFormulation : IkProblemFormulation<_Scalar>
  {
    using Base = IkProblemFormulation<_Scalar>;
    using IkIdData = typename Base::IkIdData;
    using Model = typename Base::Model;
    using Motion = typename Base::Motion;
    using DMat = typename Base::DMat;
    using DVec = typename Base::DVec;
    using Vec3 = typename Base::Vec3;
    using Vec6 = typename Base::Vec6;
    using Mat6x6 = typename Base::Mat6x6;
    using Index = typename Base::Index;
    using IndexVec = typename Base::IndexVec;

    explicit IkProblemStandardQPFormulation(
      const int nj, const int nb, const int nc_eq, const int eq_c_dim, const int ineq_c_dim)
    : Base(nj, nb, nc_eq, eq_c_dim, ineq_c_dim)
    {
      qp_constraint_dim_ = 6 * this->nb_ + this->eq_c_dim_ * this->nb_ + this->ineq_c_dim_;
      qp_var_dim_ = 6 * this->nb_ + this->ineq_c_dim_;

      Reset();
    }

    ///
    /// \brief this should only be called to completely reset the problem formulation, it will set
    /// all quantities to zero,
    ///        but problem dimensions and number of bodies and constraints won't change
    ///
    void Reset()
    {
      Base::Reset();
      // hard constraints (equality)
      A_qp_ = DMat::Zero(qp_constraint_dim_, qp_var_dim_);
      x_qp_ = DVec::Zero(qp_var_dim_);
      z_qp_ = DVec::Zero(qp_constraint_dim_);
      y_qp_ = DVec::Zero(qp_constraint_dim_);

      // soft constraints (cost)
      P_qp_ = DMat::Zero(qp_var_dim_, qp_var_dim_);
      q_qp_ = DVec::Zero(qp_var_dim_);

      // hard constraints (slack inequality)
      lb_qp_ = DVec::Zero(qp_constraint_dim_);
      ub_qp_ = DVec::Zero(qp_constraint_dim_);

      //
      x_qp_prev_ = DVec::Zero(qp_var_dim_);
      z_qp_prev_ = DVec::Zero(qp_constraint_dim_);
      y_qp_prev_ = DVec::Zero(qp_constraint_dim_);

      //
      delta_x_qp_ = DVec::Zero(qp_var_dim_);
      delta_z_qp_ = DVec::Zero(qp_constraint_dim_);
      delta_y_qp_ = DVec::Zero(qp_constraint_dim_);

      //
      delta_y_qp_plus_ = DVec::Zero(qp_constraint_dim_);
      delta_y_qp_minus_ = DVec::Zero(qp_constraint_dim_);
    };

    ///
    /// \brief problem formulation update function, should be called before runing ADMM iterations
    ///        should only be called once per ADMM solve() call, used to update equality constraints
    ///        and tracking references
    ///
    void UpdateQPADMMSolveInit(
      const Mat6x6 & H_ref,
      const Motion & v_ref,
      const std::vector<Index> & active_task_constraint_ids,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) & Ais,
      const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) & bis,
      const DVec & lb,
      const DVec & ub,
      const Model & model,
      const IkIdData & ik_id_data)
    {
      IndexVec joint_full_range = ik_id_data.joint_full_range;
      IndexVec joint_range = ik_id_data.joint_range;

      Base::UpdateReference(H_ref, v_ref);
      Base::UpdateIneqConstraints(lb, ub);
      Base::UpdateEqConstraints(active_task_constraint_ids, Ais, bis);

      lb_qp_.segment(6 * this->nb_ + this->eq_c_dim_ * this->nb_, this->ineq_c_dim_) = this->lb_;
      ub_qp_.segment(6 * this->nb_ + this->eq_c_dim_ * this->nb_, this->ineq_c_dim_) = this->ub_;

      A_qp_.block(0, 0, 6 * this->nb_, 6 * this->nb_) =
        -1.0 * DMat::Identity(6 * this->nb_, 6 * this->nb_);
      A_qp_.block(
        6 * this->nb_ + this->eq_c_dim_ * this->nb_, 6 * this->nb_, this->ineq_c_dim_,
        this->ineq_c_dim_) = DMat::Identity(this->ineq_c_dim_, this->ineq_c_dim_);

      // build A_qp, P_qp, q_qp
      for (Index idx : joint_range)
      {

        // fill in the Hi_ref and vi_ref, into P_qp_ and q_qp_
        int row_start_idx_Hi_vi = (static_cast<int>(idx) - 1) * 6;
        int col_start_idx_Hi_vi = (static_cast<int>(idx) - 1) * 6;

        P_qp_.block(row_start_idx_Hi_vi, col_start_idx_Hi_vi, 6, 6).noalias() = this->H_refs_[idx];
        q_qp_.segment(row_start_idx_Hi_vi, 6).noalias() =
          -(this->H_refs_[idx]).transpose() * (this->v_refs_[idx]).toVector();

        // fill in the Sis into A_qp_
        int row_start_idx_Si = (static_cast<int>(idx) - 1) * 6;
        int joint_nv = model.joints[idx].nv();
        int col_start_idx_Si = 6 * this->nb_ + model.joints[idx].idx_v();
        A_qp_.block(row_start_idx_Si, col_start_idx_Si, 6, joint_nv).noalias() =
          ik_id_data.joints[idx].S().matrix();

        // fill in the band diagonal frame transformations into A_qp_
        int col_start_idx_i = (static_cast<int>(idx) - 1) * 6;
        Index parent = model.parents[idx];
        // ik_id_data.oMi[idx].inverse().toActionMatrix(); //iMo

        if (parent > 0)
        {
          // ik_id_data.oMi[parent].toActionMatrix(); // oMp
          int col_start_idx_parent = (static_cast<int>(parent) - 1) * 6;

          //                                                                    |<------------------
          //                                                                    iMo
          //                                                                    ------------------>|
          //                                                                    |<---------------
          //                                                                    oMp
          //                                                                    --------------->|
          A_qp_.block(row_start_idx_Si, col_start_idx_parent, 6, 6).noalias() =
            ik_id_data.oMi[idx].inverse().toActionMatrix()
            * ik_id_data.oMi[parent].toActionMatrix();
          A_qp_.block(row_start_idx_Si, col_start_idx_i, 6, 6).noalias() = -Mat6x6::Identity();
        }
        else if (parent == 0)
        {
          A_qp_.block(row_start_idx_Si, col_start_idx_i, 6, 6).noalias() = -Mat6x6::Identity();
        }
        else
        {
          throw(std::runtime_error("[IkProblemStandardQPFormulation::UpdateQPADMMSolveInit]: "
                                   "parent id < 0, this can't happen !!!"));
        }
      }

      // build lb_qp, ub_qp
      Index c_idx_in_vec = 0;
      for (Index c_idx : this->active_task_constraint_ids_)
      {

        // fill in the Ais from motion constraints into A_qp_

        int row_start_idx_Ai = 6 * this->nb_ + (static_cast<int>(c_idx) - 1) * this->eq_c_dim_;
        int col_start_idx_Ai = (static_cast<int>(c_idx) - 1) * 6;

        A_qp_.block(row_start_idx_Ai, col_start_idx_Ai, this->eq_c_dim_, 6).noalias() =
          this->Ais_[c_idx_in_vec];

        // fill in the bis from motion constraints into ub_qp_ and lb_qp_
        int row_start_idx_boundi = 6 * this->nb_ + (static_cast<int>(c_idx) - 1) * this->eq_c_dim_;

        lb_qp_.segment(row_start_idx_boundi, this->eq_c_dim_).noalias() = this->bis_[c_idx_in_vec];
        ub_qp_.segment(row_start_idx_boundi, this->eq_c_dim_).noalias() = this->bis_[c_idx_in_vec];

        c_idx_in_vec++;
      }

      // fill part of z_qp_ for motion constriants
      z_qp_.segment(6 * this->nb_, this->eq_c_dim_ * this->nb_).noalias() =
        ub_qp_.segment(6 * this->nb_, this->eq_c_dim_ * this->nb_);
    };

    ///
    /// \brief update x_qp_, y_qp_, and z_qp_ after each ADMM solve,
    ///
    void UpdateQPADMMSolveLoop(const IkIdData & ik_id_data)
    {
      // update, log quantities from previous iterations
      x_qp_prev_ = x_qp_;
      z_qp_prev_ = z_qp_;
      y_qp_prev_ = y_qp_;

      for (Index idx : ik_id_data.joint_range)
      {
        int row_start_idx_vi = (static_cast<int>(idx) - 1) * 6;
        int row_start_idx_fi = (static_cast<int>(idx) - 1) * 6;
        int row_start_idx_yi = 6 * this->nb_ + (static_cast<int>(idx) - 1) * this->eq_c_dim_;

        // fill x_qp_
        x_qp_.segment(row_start_idx_vi, 6).noalias() = ik_id_data.vis[idx].toVector(); // vi

        // fill dual variable y_qp associated with all constraints
        y_qp_.segment(row_start_idx_fi, 6).noalias() = ik_id_data.fis[idx];               // fi
        y_qp_.segment(row_start_idx_yi, this->eq_c_dim_).noalias() = ik_id_data.yis[idx]; // yi
      }

      // fill remainder of x_qp_
      x_qp_.segment(6 * this->nb_, this->ineq_c_dim_).noalias() = ik_id_data.nu;

      // fill remainder of y_qp_
      y_qp_.segment(6 * this->nb_ + this->nb_ * this->eq_c_dim_, this->ineq_c_dim_).noalias() =
        ik_id_data.w;

      // fill part of z_qp_ related to inequality slack
      z_qp_.segment(6 * this->nb_ + this->eq_c_dim_ * this->nb_, this->ineq_c_dim_).noalias() =
        ik_id_data.z;

      // udpate deltas
      delta_x_qp_ = x_qp_ - x_qp_prev_;
      delta_y_qp_ = y_qp_ - y_qp_prev_;
      delta_z_qp_ = z_qp_ - z_qp_prev_;

      // update delta_y_qp_ plus and minus
      delta_y_qp_plus_ = delta_y_qp_.cwiseMax(0);
      delta_y_qp_minus_ = delta_y_qp_.cwiseMin(0);
    };

    int qp_constraint_dim_;
    int qp_var_dim_;
    DMat A_qp_;  // constraint matrix 'A' in OSQP problem formulation
    DVec x_qp_;  // primal decision variables in OSQP problem formulation
    DVec z_qp_;  // slack variables in OSQP problem formulation
    DVec y_qp_;  // dual variables in OSQP problem formulation
    DMat P_qp_;  // quadratic cost weight in OSQP problem formulation
    DVec q_qp_;  // linear cost in OSQP problem formulation
    DVec lb_qp_; // lower bound for primal variables in OSQP problem formulation
    DVec ub_qp_; // upper bound for primal variables in OSQP problem formulation

    DVec x_qp_prev_;
    DVec z_qp_prev_;
    DVec y_qp_prev_;

    DVec delta_x_qp_;
    DVec delta_z_qp_;
    DVec delta_y_qp_;

    DVec delta_y_qp_plus_;  // copy of 'delta_y_qp_' with all negative coefficients set to 0
    DVec delta_y_qp_minus_; // copy of 'delta_y_qp_' with all positive coefficients set to 0

  }; // struct IkProblemStandardQPFormulation

} // namespace loik
