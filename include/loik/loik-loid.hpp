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
    // using Base = IkIdSolverBaseTpl<_Scalar>;
    typedef typename Base::Scalar Scalar;
    // using Scalar = typename Base::Scalar;
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

    FirstOrderLoikTpl(const int max_iter, const Scalar& tol_abs, const Scalar& tol_rel, const Scalar& tol_primal_inf, const Scalar& tol_dual_inf,
                   const Scalar& rho, const Scalar& mu, const Scalar& mu_equality_scale_factor, const ADMMPenaltyUpdateStrat& mu_update_strat, 
                   const int num_eq_c, const int eq_c_dim, 
                   const Model& model, IkIdData& ik_id_data,
                   const bool warm_start,
                   const bool verbose,  const bool logging) 
        : Base(max_iter, tol_abs, tol_rel, tol_primal_inf, tol_dual_inf, 
               rho, mu, mu_equality_scale_factor, mu_update_strat, 
               verbose, logging), 
          model_(model),
          ik_id_data_(ik_id_data),
          problem_(model.njoints, model.njoints - 1, num_eq_c, eq_c_dim, model.nv),
          nj_(model.njoints),
          nb_(model.njoints - 1),
          nv_(model.nv),
          warm_start_(warm_start),
          loik_solver_info_(max_iter)
    {
        // initialize helper quantities
        joint_full_range_ = ik_id_data_.joint_full_range; // [0, nj - 1]
        joint_range_ = ik_id_data_.joint_range;           // [1, nj - 1]


        ResetSolver();
    };


    ///
    /// \brief Reset the diff IK solver
    ///
    void ResetSolver();


    ///
    /// \brief Initial forward pass, to propagate forward kinematics. 
    ///
    void FwdPassInit(const DVec& q)
    {
        // std::cout << "*******************FwdPassInit*******************" << std::endl;

        LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        for (const auto& idx : joint_range_) {
            const JointModel& jmodel = model_.joints[idx];
            JointData& jdata = ik_id_data_.joints[idx];
            Index parent = model_.parents[idx];

            // computes "M"s for each joint, i.e. displacement in current joint frame caused by 'self.q_'
            jmodel.calc(jdata, q);
            ik_id_data_.liMi[idx] = model_.jointPlacements[idx] * jdata.M();
            ik_id_data_.oMi[idx] = ik_id_data_.oMi[parent] * ik_id_data_.liMi[idx];

            // const auto& Si = jdata.S();
            // const auto Si_mat = jdata.S().matrix();

            // std::cout << "Si: " << Si << std::endl;
            // std::cout << "Si matrix: " << Si_mat << std::endl;
        }

        LOIK_EIGEN_MALLOC_ALLOWED();

        // std::cout << " " << std::endl;
    };


    ///
    /// \brief LOIK first forward pass
    ///
    void FwdPass1()
    {
        LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        for (const auto& idx : joint_range_) {
            const JointModel& jmodel = model_.joints[idx];
            // JointData& jdata = ik_id_data_.joints[idx];
            // Index parent = model_.parents[idx];

            int joint_nv = jmodel.nv();
            int joint_idx_v = jmodel.idx_v();

            // build Ris and ris TODO: Ris only need to be computed once for constant mu
            ik_id_data_.Ris[idx] = mu_ineq_ * DMat::Identity(joint_nv, joint_nv);
            const auto& wi = ik_id_data_.w.segment(joint_idx_v, joint_nv);
            const auto& zi = ik_id_data_.z.segment(joint_idx_v, joint_nv);
            ik_id_data_.ris[idx].noalias() = wi - mu_ineq_ * zi;

            const Mat6x6& H_ref = problem_.H_refs_[idx];
            const Motion& v_ref = problem_.v_refs_[idx];
            ik_id_data_.His[idx].noalias() = this->rho_ * DMat::Identity(6, 6) + H_ref;

            // LOIK_EIGEN_MALLOC_NOT_ALLOWED();
            ik_id_data_.pis[idx].noalias() = - this->rho_ * ik_id_data_.vis_prev[idx].toVector() - H_ref.transpose() * v_ref.toVector();
            // LOIK_EIGEN_MALLOC_ALLOWED();
        }

        
        Index c_vec_id = 0;
        for (const auto& c_id : problem_.active_task_constraint_ids_) {
            const Mat6x6& Ai = problem_.Ais_[c_vec_id];
            const Vec6& bi = problem_.bis_[c_vec_id];

            const DVec& yi = ik_id_data_.yis[c_id];
            ik_id_data_.His[c_id].noalias() += mu_eq_ * Ai.transpose() * Ai;
            ik_id_data_.pis[c_id].noalias() += Ai.transpose() * yi - mu_eq_ * Ai.transpose() * bi;

            c_vec_id++;
        }

        LOIK_EIGEN_MALLOC_ALLOWED();

    };


    ///
    /// \brief LOIK first packward pass
    ///
    void BwdPass()
    {
        // LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        // loop over joint range in reverse
        for (auto it = joint_range_.rbegin(); it != joint_range_.rend(); ++it) {
            Index idx = *it;

            // JointModel& jmodel = model_.joints[idx];
            JointData& jdata = ik_id_data_.joints[idx];
            Index parent = model_.parents[idx];

            // int joint_nv = jmodel.nv();
            const SE3& liMi = ik_id_data_.liMi[idx];
            const Mat6x6& Hi = ik_id_data_.His[idx];
            const Vec6& pi = ik_id_data_.pis[idx];
            const DMat& Si = jdata.S().matrix();      // TODO: this will cause memory allocation
            const DMat& Ri = ik_id_data_.Ris[idx];
            const DVec& ri = ik_id_data_.ris[idx];
            
            
            ik_id_data_.Dis[idx].noalias() = Ri + Si.transpose() * Hi * Si;         // TODO: this will cause memory allocation
            ik_id_data_.Di_invs[idx].noalias() = ik_id_data_.Dis[idx].inverse();    // TODO: this will cause memory allocation
            const DMat& Di_inv = ik_id_data_.Di_invs[idx];
            ik_id_data_.Pis[idx].noalias() = DMat::Identity(6, 6) - Hi * Si * Di_inv * Si.transpose();      // TODO: this will cause memory allocation
            const Mat6x6& Pi = ik_id_data_.Pis[idx];

            // ik_id_data_.His[parent].noalias() += liMi.toDualActionMatrix() * (Pi * Hi) * liMi.inverse().toActionMatrix();
            ik_id_data_.His[parent].noalias() += liMi.toDualActionMatrix() * (Pi * Hi) * liMi.toActionMatrixInverse();
            ik_id_data_.pis[parent].noalias() += liMi.toDualActionMatrix() * (Pi * pi - Hi * Si * Di_inv * ri);         // TODO: this will cause memory allocation
            

        }

        // LOIK_EIGEN_MALLOC_ALLOWED();
        
    };


    ///
    /// \brief LOIK first packward pass optimized
    ///
    void BwdPassOptimized()
    {
        

        // loop over joint range in reverse
        for (auto it = joint_range_.rbegin(); it != joint_range_.rend(); ++it) {
            LOIK_EIGEN_MALLOC_NOT_ALLOWED();
            Index idx = *it;

            const JointModel& jmodel = model_.joints[idx];
            JointData& jdata = ik_id_data_.joints[idx];
            Index parent = model_.parents[idx];

            // int joint_nv = jmodel.nv();
            const SE3& liMi = ik_id_data_.liMi[idx];
            const Mat6x6& Hi = ik_id_data_.His[idx];
            const Vec6& pi = ik_id_data_.pis[idx];
            
            const DMat& Ri = ik_id_data_.Ris[idx];
            DVec& ri = ik_id_data_.ris[idx];

            jmodel.calc_aba(jdata.derived(), 
                            Ri.diagonal(), 
                            Hi, 
                            parent > 0);

            ik_id_data_.His[parent].noalias() += pinocchio::impl::internal::SE3actOn<Scalar>::run(liMi, Hi);
            
            LOIK_EIGEN_MALLOC_ALLOWED();

            ri.noalias() += jdata.S().transpose() * pi;


            // pi.noalias() -= jdata.UDinv() * ri;
            ik_id_data_.pis[parent].noalias() += liMi.toDualActionMatrix() * (pi - jdata.UDinv() * ri);
            // ik_id_data_.Dis[idx].noalias() = Ri + Si.transpose() * Hi * Si;         // TODO: this will cause memory allocation
            // ik_id_data_.Di_invs[idx].noalias() = ik_id_data_.Dis[idx].inverse();    // TODO: this will cause memory allocation
            // const DMat& Di_inv = ik_id_data_.Di_invs[idx];
            // ik_id_data_.Pis[idx].noalias() = DMat::Identity(6, 6) - Hi * Si * Di_inv * Si.transpose();      // TODO: this will cause memory allocation
            // const Mat6x6& Pi = ik_id_data_.Pis[idx];

            // ik_id_data_.His[parent].noalias() += liMi.toDualActionMatrix() * (Pi * Hi) * liMi.inverse().toActionMatrix();
            // ik_id_data_.His[parent].noalias() += liMi.toDualActionMatrix() * (Pi * Hi) * liMi.toActionMatrixInverse();
            // ik_id_data_.pis[parent].noalias() += liMi.toDualActionMatrix() * (Pi * pi - Hi * Si * Di_inv * ri);         // TODO: this will cause memory allocation
            

        }

        
        
    };


    ///
    /// \brief LOIK second forward pass
    ///
    void FwdPass2()
    {
        // std::cout << "****************FwdPass2******************" << std::endl;

        // LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        for (const auto& idx : joint_range_) {

            // std::cout << "idx: " << idx << std::endl;

            const JointModel& jmodel = model_.joints[idx];
            JointData& jdata = ik_id_data_.joints[idx];
            Index parent = model_.parents[idx];

            int joint_nv = jmodel.nv();
            int joint_idx_v = jmodel.idx_v();

            const DMat& Di_inv = ik_id_data_.Di_invs[idx];
            const auto Si = jdata.S().matrix();
            const Mat6x6& Hi = ik_id_data_.His[idx];
            const Vec6& pi = ik_id_data_.pis[idx];
            const DVec& ri = ik_id_data_.ris[idx];
            const SE3& liMi = ik_id_data_.liMi[idx];
            const Motion vi_parent = liMi.actInv(ik_id_data_.vis[parent]); // ith joint's parent spatial velocity in joint i's local frame

        

            ik_id_data_.nu.segment(joint_idx_v, joint_nv).noalias() = - Di_inv * (Si.transpose() * (Hi * vi_parent.toVector() + pi) + ri);

            // std::cout << "///////////////////////" << std::endl;
            // std::cout << "Di_inv: " << Di_inv << std::endl;
            // std::cout << "Si: " << Si << std::endl;
            // std::cout << "Hi: " << Hi << std::endl;
            // std::cout << "vi_parent: " << vi_parent << std::endl;
            // std::cout << "pi: " << pi.transpose() << std::endl;
            // std::cout << "ri: " << ri.transpose() << std::endl;
            
            // std::cout << "nu_i: " << (- Di_inv * (Si.transpose() * (Hi * vi_parent.toVector()) + ri)).transpose() << std::endl;

            ik_id_data_.Si_nui_s[idx] = Motion(Si * ik_id_data_.nu.segment(joint_idx_v, joint_nv));
            ik_id_data_.vis[idx] = vi_parent + ik_id_data_.Si_nui_s[idx];

            ik_id_data_.fis[idx].noalias() = Hi * ik_id_data_.vis[idx].toVector() + pi;

        }

        // LOIK_EIGEN_MALLOC_ALLOWED();

        // std::cout << "nu: " << ik_id_data_.nu.transpose() << std::endl;

        // std::cout << "*****************************************" << std::endl;
    };

        ///
    /// \brief LOIK second forward pass
    ///
    void FwdPass2Optimized()
    {
        // std::cout << "****************FwdPass2******************" << std::endl;

        // LOIK_EIGEN_MALLOC_NOT_ALLOWED();

        for (const auto& idx : joint_range_) {

            // std::cout << "idx: " << idx << std::endl;

            const JointModel& jmodel = model_.joints[idx];
            JointData& jdata = ik_id_data_.joints[idx];
            Index parent = model_.parents[idx];

            int joint_nv = jmodel.nv();
            int joint_idx_v = jmodel.idx_v();

            // const DMat& Di_inv = ik_id_data_.Di_invs[idx];
            const Mat6x6& Hi = ik_id_data_.His[idx];
            const Vec6& pi = ik_id_data_.pis[idx];
            const DVec& ri = ik_id_data_.ris[idx];
            const SE3& liMi = ik_id_data_.liMi[idx];
            const Motion vi_parent = liMi.actInv(ik_id_data_.vis[parent]); // ith joint's parent spatial velocity in joint i's local frame


            ik_id_data_.nu.segment(joint_idx_v, joint_nv).noalias() = - jdata.UDinv().transpose() * vi_parent.toVector() - jdata.Dinv()*ri;

            // std::cout << "///////////////////////" << std::endl;
            // std::cout << "Di_inv: " << Di_inv << std::endl;
            // std::cout << "Si: " << Si << std::endl;
            // std::cout << "Hi: " << Hi << std::endl;
            // std::cout << "vi_parent: " << vi_parent << std::endl;
            // std::cout << "pi: " << pi.transpose() << std::endl;
            // std::cout << "ri: " << ri.transpose() << std::endl;
            
            // std::cout << "nu_i: " << (- Di_inv * (Si.transpose() * (Hi * vi_parent.toVector()) + ri)).transpose() << std::endl;

            ik_id_data_.Si_nui_s[idx] = Motion(jdata.S() * ik_id_data_.nu.segment(joint_idx_v, joint_nv));
            ik_id_data_.vis[idx] = vi_parent + ik_id_data_.Si_nui_s[idx];

            ik_id_data_.fis[idx].noalias() = Hi * ik_id_data_.vis[idx].toVector() + pi;

        }

    };

    ///
    /// \brief Box projection of primal and slack composite quantites 
    ///
    void BoxProj()
    {
        LOIK_EIGEN_MALLOC_NOT_ALLOWED();
        // update slack
        ik_id_data_.z.noalias() = problem_.ub_.cwiseMin(problem_.lb_.cwiseMax(ik_id_data_.nu + (1.0 / mu_ineq_) * ik_id_data_.w)) ;
        LOIK_EIGEN_MALLOC_ALLOWED();
    };


    ///
    /// \brief ADMM dual variable updates
    ///
    void DualUpdate()
    {
        // LOIK_EIGEN_MALLOC_NOT_ALLOWED();
        // update dual variables associated with motion constraints 'ik_id_data_.yis' 
        Index c_vec_id = 0;
        for (const auto& c_id : problem_.active_task_constraint_ids_) {
            const Mat6x6& Ai = problem_.Ais_[c_vec_id];
            const Vec6& bi = problem_.bis_[c_vec_id];
            const Motion& vi = ik_id_data_.vis[c_id];

            // const auto temp = (Ai * vi.toVector() - bi);

            ik_id_data_.yis[c_id].noalias() += mu_eq_ * (Ai * vi.toVector() - bi);
            
            c_vec_id++;
        }

        // update dual vairables associated with inequality slack induced equality constraints 'ik_id_data_.w'
        ik_id_data_.w.noalias() += mu_ineq_ * (ik_id_data_.nu - ik_id_data_.z);

        // LOIK_EIGEN_MALLOC_ALLOWED();
    };


    ///
    /// \brief unit test utility function
    ///
    void UpdateQPADMMSolveLoopUtility()
    {
        // update standard qp formulation using primal dual variables from current iter
        problem_.UpdateQPADMMSolveLoop(ik_id_data_);
    };


    ///
    /// \brief Compute solver residuals
    ///
    void ComputeResiduals()
    {
        if (this->verbose_) {
            std::cout << "****************ComputeResiduals******************" << std::endl;
        }
        const int m = problem_.eq_c_dim_;

        // compute primal residual
        Index c_vec_id = 0;
        for (const auto& c_id : problem_.active_task_constraint_ids_) {
            const Mat6x6& Ai = problem_.Ais_[c_vec_id];
            const Vec6& bi = problem_.bis_[c_vec_id];
            const Motion& vi = ik_id_data_.vis[c_id];

            primal_residual_vec_.segment(m * (static_cast<int>(c_id) - 1), m) = Ai * vi.toVector() - bi;

            c_vec_id++;
        }


        primal_residual_vec_.segment(m * nb_, nv_) = ik_id_data_.nu - ik_id_data_.z;
        this->primal_residual_ = primal_residual_vec_.template lpNorm<Eigen::Infinity>();
        primal_residual_task_ = primal_residual_vec_.segment(0, m * nb_).template lpNorm<Eigen::Infinity>();
        primal_residual_slack_ = primal_residual_vec_.segment(m * nb_, nv_).template lpNorm<Eigen::Infinity>();

        // std::cout << "primal_residual_vec: " << primal_residual_vec_.transpose() << std::endl;
        // std::cout << "primal residual check: " << primal_residual_vec_.template lpNorm<Eigen::Infinity>() << std::endl;

        // DEBUG testing primal kinematics residual
        if (this->verbose_) {
            DVec primal_residual_from_qp = problem_.A_qp_ * problem_.x_qp_ - problem_.z_qp_;

            std::cout << "primal residual kinematics constraint violation from QP:" 
                    << (primal_residual_from_qp.segment(0, 6 * nb_)).template lpNorm<Eigen::Infinity>() 
                    << std::endl;

            std::cout << "primal residual: " << this->primal_residual_ << std::endl;
            std::cout << "primal residual task: " << primal_residual_task_ << std::endl;
            std::cout << "primal residual slack: " << primal_residual_slack_ << std::endl;
        }


        // compute dual residual 
        for (auto it = joint_range_.rbegin(); it != joint_range_.rend(); ++it) {
            Index idx = *it;

            const Mat6x6& H_ref = problem_.H_refs_[idx];
            const Motion& v_ref = problem_.v_refs_[idx];
            const Motion& vi = ik_id_data_.vis[idx];
            // const DVec& yi = ik_id_data_.yis[idx];
            const Vec6& fi = ik_id_data_.fis[idx];
            const SE3& liMi = ik_id_data_.liMi[idx];
            Index parent = model_.parents[idx];

            int row_start_idx = (static_cast<int>(idx) - 1) * 6;
            int row_start_idx_parent = 0;

            if (static_cast<int>(parent) > 0) {
                row_start_idx_parent = (static_cast<int>(parent) - 1) * 6;
                dual_residual_vec_.segment(row_start_idx, 6).noalias() += H_ref * vi.toVector()  - H_ref * v_ref.toVector() - fi;
                dual_residual_vec_.segment(row_start_idx_parent, 6).noalias() += liMi.toDualActionMatrix() * fi;
            } else {
                row_start_idx_parent = 0;
                dual_residual_vec_.segment(row_start_idx, 6).noalias() += H_ref * vi.toVector()  - H_ref * v_ref.toVector() - fi;
            }

            const JointModel& jmodel = model_.joints[idx];
            const JointData& jdata = ik_id_data_.joints[idx];
            const int row_start_idx_wi = jmodel.idx_v();

            const auto& wi = ik_id_data_.w.segment(row_start_idx_wi, jmodel.nv());
            const auto Si = jdata.S().matrix();

            dual_residual_vec_.segment(6 * nb_ + row_start_idx_wi, jmodel.nv()).noalias() = Si.transpose() * fi + wi;


        }

        c_vec_id = 0;
        for (const auto& c_id : problem_.active_task_constraint_ids_) {
            const Mat6x6& Ai = problem_.Ais_[c_vec_id];
            const Vec6& yi = ik_id_data_.yis[c_id];

            int row_start_idx = (static_cast<int>(c_id) - 1) * 6;

            dual_residual_vec_.segment(row_start_idx, 6).noalias() += Ai.transpose() * yi;

            c_vec_id ++;
        }


        dual_residual_vec_ = problem_.P_qp_ * problem_.x_qp_ + problem_.q_qp_ + problem_.A_qp_.transpose() * problem_.y_qp_;

        // 
        dual_residual_prev_ = this->dual_residual_;
        dual_residual_v_prev_ = dual_residual_v_;
        dual_residual_nu_prev_ = dual_residual_nu_;

        this->dual_residual_ = dual_residual_vec_.template lpNorm<Eigen::Infinity>();
        dual_residual_v_ = (dual_residual_vec_.segment(0, 6 * nb_)).template lpNorm<Eigen::Infinity>();
        dual_residual_nu_ = (dual_residual_vec_.segment(6 * nb_, nv_)).template lpNorm<Eigen::Infinity>();

        delta_dual_residual_ = this->dual_residual_ - dual_residual_prev_;
        delta_dual_residual_v_ = dual_residual_v_ - dual_residual_v_prev_;
        delta_dual_residual_nu_ = dual_residual_nu_ - dual_residual_nu_prev_;

        if (this->verbose_) {
            std::cout << "dual residual: " << this->dual_residual_ << std::endl;
            std::cout << "dual residual v: " << dual_residual_v_ << std::endl;
            std::cout << "dual residual nu: " << dual_residual_nu_ << std::endl;

            std::cout << "normInf delta_x_qp: " << problem_.delta_x_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "normInf delta_z_qp: " << problem_.delta_z_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "normInf delta_y_qp: " << problem_.delta_y_qp_.template lpNorm<Eigen::Infinity>() << std::endl;
        }

        if (this->verbose_) {
            DVec dual_residual_from_qp = problem_.P_qp_ * problem_.x_qp_ + problem_.q_qp_ + problem_.A_qp_.transpose() * problem_.y_qp_;

            std::cout << "dual residual from QP:" 
                      << dual_residual_from_qp.template lpNorm<Eigen::Infinity>() 
                      << std::endl;

            std::cout << "dual residual v from QP:" 
                      << (dual_residual_from_qp.segment(0, 6 * nb_)).template lpNorm<Eigen::Infinity>() 
                      << std::endl;
            
            std::cout << "dual residual nu from QP:" 
                      << (dual_residual_from_qp.segment(6 * nb_, nv_)).template lpNorm<Eigen::Infinity>() 
                      << std::endl;
        }


    };


    ///
    /// \brief Check primal and dual convergence
    ///
    void CheckConvergence()
    {   
        // update primal residual tolerance 
        // TODO: move 'problem_.A_qp_ * problem_.x_qp_' to FwdPass2()
        this->tol_primal_ = this->tol_abs_ 
                            + this->tol_rel_ * std::max((problem_.A_qp_ * problem_.x_qp_).template lpNorm<Eigen::Infinity>(), 
                                                        (problem_.z_qp_).template lpNorm<Eigen::Infinity>());
        
        // update dual residual tolerance
        // TODO: move 'problem_.P_qp_ @ problem_.x_qp_' to FwdPass2()
        //            'problem_.A_qp_.transpose() * problem_.y_qp_' to DualUpdate()
        this->tol_dual_ = this->tol_abs_ 
                          + this->tol_rel_ * std::max(std::max((problem_.P_qp_ * problem_.x_qp_).template lpNorm<Eigen::Infinity>(), 
                                                      (problem_.A_qp_.transpose() * problem_.y_qp_).template lpNorm<Eigen::Infinity>()), 
                                                      (problem_.q_qp_).template lpNorm<Eigen::Infinity>());

        // check convergence
        if ( (this->primal_residual_ < this->tol_primal_) && (this->dual_residual_ < this->tol_dual_) ) {
            this->converged_ = true;

            if (this->verbose_) {
                std::cerr << "[FirstOrderLoik::CheckConvergence]: converged in " << this->iter_ << "iterations !!!" << std::endl;
            }
        }

    };


    ///
    /// \brief Check primal and dual feasibility
    ///
    void CheckFeasibility()
    {
        
        // check for primal infeasibility
        bool primal_infeasibility_cond_1 = (problem_.A_qp_.transpose() * problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>() 
                                           <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();
        
        bool primal_infeasibility_cond_2 = (problem_.ub_qp_ .transpose() * problem_.delta_y_qp_plus_ + problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_).value()
                                           <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();


        if (this->verbose_) {
            std::cout << "****************************CheckFeasibility****************************" << std::endl;
        
            std::cout << "primal_infeasibility_cond_1: " << primal_infeasibility_cond_1 << std::endl;
            std::cout << "delta_y_qp normInf: " << (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "check lhs: " << ((problem_.A_qp_.transpose() * problem_.delta_y_qp_)).template lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "check lhs kinematics: " << ((problem_.A_qp_.transpose() * problem_.delta_y_qp_).segment(0, 6 * nb_)).template lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "check lhs slack: " << ((problem_.A_qp_.transpose() * problem_.delta_y_qp_).segment(6 * nb_, nv_)).template lpNorm<Eigen::Infinity>() << std::endl;
            
            std::cout << "check rhs: " << this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>() << std::endl;

            std::cout << "primal_infeasibility_cond_2: " << primal_infeasibility_cond_2 << std::endl;
            std::cout << "check lhs: " << (problem_.ub_qp_ .transpose() * problem_.delta_y_qp_plus_ + problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_).value() << std::endl;
            std::cout << "check rhs: " << this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>() << std::endl;
            
        }
        

        if (primal_infeasibility_cond_1 && primal_infeasibility_cond_2) {
            this->primal_infeasible_ = true;
            if (this->verbose_) {
                std::cerr << "WARNING [FirstOrderLoik::CheckFeasibility]: IK problem is primal infeasible !!!" << std::endl;
            }
        }

        // check for dual infeasibility 
        bool dual_infeasibility_cond_1 = (problem_.P_qp_ * problem_.delta_x_qp_). template lpNorm<Eigen::Infinity>()
                                         <= this->tol_dual_inf_ * (problem_.delta_x_qp_). template lpNorm<Eigen::Infinity>(); 
        bool dual_infeasibility_cond_2 = (problem_.q_qp_.transpose() * problem_.delta_x_qp_).value() 
                                         <= this->tol_dual_inf_ * (problem_.delta_x_qp_). template lpNorm<Eigen::Infinity>();

        if (dual_infeasibility_cond_1 && dual_infeasibility_cond_2) {
            bool dual_infeasibility_cond_3 = ((problem_.A_qp_ * problem_.delta_x_qp_).array() >= -this->tol_dual_inf_ * (problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>()).all();
            bool dual_infeasibility_cond_4 = ((problem_.A_qp_ * problem_.delta_x_qp_).array() <= this->tol_dual_inf_ * (problem_.delta_x_qp_).template lpNorm<Eigen::Infinity>()).all();

            if (dual_infeasibility_cond_3 && dual_infeasibility_cond_4) {
                this->dual_infeasible_ = true;
                if (this->verbose_) {
                    std::cerr << "WARNING [FirstOrderLoik::CheckFeasibility]: IK problem is dual infeasible !!!" << std::endl;
                }
            }

        }
    };


    ///
    /// \brief Update ADMM penalty mu 
    ///
    void UpdateMu()
    { 
        // std::cout << "***************UpdateMu****************" << std::endl;
        if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::DEFAULT) {
            // update mu by threasholding primal and dual residual ratio
            if (this->primal_residual_ > 10 * this->dual_residual_) {
                this->mu_ *= 10;

                mu_eq_ = this->mu_equality_scale_factor_ * this->mu_;
                mu_ineq_ = this->mu_;
                // std::cout << "mu: " << this->mu_ << std::endl;
                return;
            } else if (this->dual_residual_ > 10 * this->primal_residual_) {
                this->mu_ *= 0.1;

                mu_eq_ = this->mu_equality_scale_factor_ * this->mu_;
                mu_ineq_ = this->mu_;
                // std::cout << "mu: " << this->mu_ << std::endl;
                return;
            } else { 
                // no update to 'mu_' reutrn
                // std::cout << "mu: " << this->mu_ << std::endl;
                return;
            }
        } else if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::OSQP) {
            // using OSQP strategy
            throw(std::runtime_error("[FirstOrderLoik::UpdateMu]: mu update strategy OSQP not yet implemented"));
        } else if (this->mu_update_strat_ == ADMMPenaltyUpdateStrat::MAXEIGENVALUE) {
            // use max eigen value strategy 
            throw(std::runtime_error("[FirstOrderLoik::UpdateMu]: mu update strategy MAXEIGENVALUE not yet implemented"));
        } else {
            throw(std::runtime_error("[FirstOrderLoik::UpdateMu]: mu update strategy not supported"));
        }
    };

    ///
    /// \brief when infeasibility is detected, run tail solve so primal residual converges to something. 
    ///        This gives theoretical guarantee that the solution (unprojected) converges the closest 
    ///        feasible solution. 
    ///
    void InfeasibilityTailSolve();

    ///
    /// \brief compute primal residual final 
    ///
    void ComputePrimalResidualFinal()
    {

    };


    ///
    /// \brief Initialize the problem to be solved.
    ///
    /// \param[in] q                               current generalized configuration  (DVec)
    /// \param[in] H_ref                           Cost weight for tracking reference (DMat)
    /// \param[in] v_ref                           reference spatial velocity (DVec)
    /// \param[in] active_task_constraint_ids      vector of joint ids where equality constraints are present (std::vector)
    /// \param[in] Ais                             vector of equality constraint matrix (std::vector)
    /// \param[in] bis                             vector of equality constraint targets (std::vector)
    /// \param[in] lb                              joint velocity lower bounds (DVec)
    /// \param[in] ub                              joint velocity upper bounds (DVec)
    /// \param[out] this->ik_id_data_.z            projected joint velocities onto the box constraint set
    ///
    void SolveInit(const DVec& q, 
                   const Mat6x6& H_ref, const Motion& v_ref, 
                   const std::vector<Index>& active_task_constraint_ids, const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)& Ais, const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6)& bis, 
                   const DVec& lb, const DVec& ub);

    
    ///
    /// \brief Solve the constrained differential IK problem, just the main loop
    ///
    void Solve();


    ///
    /// \brief Stand alone Solve, solves the constrained differential IK problem.
    ///
    /// \param[in] q                               current generalized configuration  (DVec)
    /// \param[in] H_ref                           Cost weight for tracking reference (DMat)
    /// \param[in] v_ref                           reference spatial velocity (DVec)
    /// \param[in] active_task_constraint_ids      vector of joint ids where equality constraints are present (std::vector)
    /// \param[in] Ais                             vector of equality constraint matrix (std::vector)
    /// \param[in] bis                             vector of equality constraint targets (std::vector)
    /// \param[in] lb                              joint velocity lower bounds (DVec)
    /// \param[in] ub                              joint velocity upper bounds (DVec)
    /// \param[out] this->ik_id_data_.z            projected joint velocities onto the box constraint set
    ///
    void Solve(const DVec& q, 
               const Mat6x6& H_ref, const Motion& v_ref, 
               const std::vector<Index>& active_task_constraint_ids, const PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6)& Ais, const PINOCCHIO_ALIGNED_STD_VECTOR(Vec6)& bis, 
               const DVec& lb, const DVec& ub);

    
    
    inline DVec get_primal_residual_vec() const { return primal_residual_vec_; };
    inline DVec get_dual_residual_vec() const { return dual_residual_vec_; };
    inline Scalar get_dual_residual_v() const { return dual_residual_v_; };
    inline Scalar get_dual_residual_nu() const { return dual_residual_nu_; };

    /// Debug utilities 
    inline DVec get_delta_y_qp() const { return problem_.delta_y_qp_; };
    inline Scalar get_delta_y_qp_inf_norm() const { return problem_.delta_y_qp_.template lpNorm<Eigen::Infinity>(); };
    inline DVec get_A_qp_delta_y_qp() const { return problem_.A_qp_ * problem_.delta_y_qp_; };
    inline Scalar get_A_qp_T_delta_y_qp_inf_norm() const { return (problem_.A_qp_.transpose() * problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>(); };
    inline Scalar get_ub_qp_T_delta_y_qp_plus() const { return (problem_.ub_qp_.transpose() * problem_.delta_y_qp_plus_).value(); };
    inline Scalar get_lb_qp_T_delta_y_qp_minus() const { return (problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_).value(); };

    bool get_primal_infeasibility_cond_1() const
    { 
        return (problem_.A_qp_.transpose() * problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>() 
               <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();
    };

    bool get_primal_infeasibility_cond_2() const 
    {
        return (problem_.ub_qp_ .transpose() * problem_.delta_y_qp_plus_ + 
                problem_.lb_qp_.transpose() * problem_.delta_y_qp_minus_).value()
               <= this->tol_primal_inf_ * (problem_.delta_y_qp_).template lpNorm<Eigen::Infinity>();
    };



protected:
    
    const Model& model_;
    IkIdData& ik_id_data_;

    ProblemFormulation problem_;

    // ADMM solver specific quantities
    int tail_solve_iter_;                             // tail solve iteration index
    Scalar primal_residual_kinematics_;               // primal residual of just the forward kinematics equality constraints
    Scalar primal_residual_task_;                     // primal residual of just the task equality constraints
    Scalar primal_residual_slack_;                    // primal residual of just the inequality induced slack equality constraints
    DVec primal_residual_vec_;                        // utility vector for primal residual calculation 

    Scalar dual_residual_prev_;
    Scalar delta_dual_residual_; 
    Scalar dual_residual_v_;                          // dual residual of just the dual feasibility condition wrt v 
    Scalar dual_residual_v_prev_;                     
    Scalar delta_dual_residual_v_;
    Scalar dual_residual_nu_;                         // dual residual of just the dual feasibility condition wrt nu
    Scalar dual_residual_nu_prev_;
    Scalar delta_dual_residual_nu_;                 
    DVec dual_residual_vec_;                          // utility vector for dual residual calculation

    Scalar mu_eq_;                                    // ADMM penalty for equality constraints
    Scalar mu_ineq_;                                  // ADMM penalty for inequality constraints 

    // solver helper quantities
    int nj_;                                          // number of joints in the model_
    int nb_;                                          // number of bodies in the model_, 'nb_ = nj_ - 1'
    int nv_;                                          // dimension of nu_ (q_dot)
    IndexVec joint_full_range_;               // index of full joint range, [0, njoints - 1]
    IndexVec joint_range_;                    // index of joint range excluding the world/universe [1, njoints - 1]

    // warm_start flag
    bool warm_start_;

    // solver info logging struct 
    LoikSolverInfo loik_solver_info_;
    
};

} // namespace loik

#include "loik/loik-loid.hxx"

#if LOIK_ENABLE_TEMPLATE_INSTANTIATION
  #include "loik/loik-loid.txx"
#endif // LOIK_ENABLE_TEMPLATE_INSTANTIATION
