#pragma once

#include "loik/fwd.hpp"
#include "loik/loik-loid-data-optimized.hpp"

#include <pinocchio/container/boost-container-limits.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <iostream>

namespace loik {

  namespace utils {
    using ptree = boost::property_tree::ptree;

    ///
    /// \brief parse ptree to Eigen vector like object
    ///
    template <typename VecLike>
    void PtreeToVec(const ptree& pt, const VecLike& vec);

    ///
    /// \brief parse ptree to Eigen matrix like object
    ///
    template <typename MatLike>
    void PtreeToMat(const ptree& pt, const MatLike& mat);

    ///
    /// \brief parse ptree to data object
    ///
    template <typename T>
    void PtreeToDataObj(const ptree& pt, const T& data_obj);

    ///
    /// \brief parse ptree to `std::vector<T>` container-like object
    ///
    template <typename T, template<typename...> class Container>
    void PtreeToContainerOfObj(const ptree& pt, const Container<T>& container);


  } // namespace utils

  template <typename _Scalar>
  struct DiffIKProblem {
    using ptree = boost::property_tree::ptree;
    typedef IkIdDataTypeOptimizedTpl<_Scalar> IkIdDataTypeOptimized;
    IKID_DATA_TYPEDEF_TEMPLATE(IkIdDataTypeOptimized);

    std::string name;

    // solver hyper params
    int max_iter;
    Scalar tol_abs;
    Scalar tol_rel;
    Scalar tol_primal_inf;
    Scalar tol_dual_inf;
    Scalar tol_tail_solve;
    Scalar rho;
    Scalar mu0;
    Scalar mu_equality_scale_factor;
    int num_eq_c;
    int eq_c_dim = 6;
    bool warm_start = false;
    bool verbose = false;
    bool logging = false;


    // problem definitions
    DVec q;

    PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) H_refs;
    PINOCCHIO_ALIGNED_STD_VECTOR(Motion) v_refs;
    std::vector<Index> active_task_constraint_ids;
    PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) Ais;
    PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) bis;
    DVec lb;
    DVec ub;


    // solution info
    bool converged;
    bool primal_infeasible;
    bool dual_infeasible;
    Scalar primal_residual;
    Scalar dual_residual;
    Scalar mu;
    int n_iter;
    int n_tail_solve_iter;
    PINOCCHIO_ALIGNED_STD_VECTOR(Motion) vis;
    PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) yis;
    DVec w;
    DVec z;
    

    ///
    /// \brief default constructor
    ///
    DiffIKProblem() {};


  }; // struct DiffIKProblem


  template <typename _Scalar>
  struct SequenceDiffIKProblems {
    using ptree = boost::property_tree::ptree;
    typedef DiffIKProblem<_Scalar> Problem;
    using Index = typename Problem::Index;
    using DVec = typename Problem::DVec;
    using Vec6 = typename Problem::Vec6;
    using Mat6x6 = typename Problem::Mat6x6;
    using Motion = typename Problem::Motion;
    template <typename T>
    using pin_aligned_vec = pinocchio::container::aligned_vector<T>;

    std::vector<Problem> problem_sequence;

    ///
    /// \brief default constructor 
    ///
    SequenceDiffIKProblems() { Reset(); };


    ///
    /// \brief clear problem_sequence
    ///
    inline void Reset() { problem_sequence.clear(); };


    ///
    /// \brief build sequence of diff IK problems from json 
    ///
    void LoadProblemsFromJson(const std::string& file_name);

  }; // SequenceDiffIKProblems



} // namespace loik 