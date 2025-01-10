//
// Copyright (c) 2024 INRIA
//

#include "loik/fwd.hpp"
#include "loik/loik-loid.hpp"
#include "loik/loik-loid-optimized.hpp"

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <pinocchio/utils/check.hpp>
#include <pinocchio/utils/timer.hpp>

#include <Eigen/Eigenvalues>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

#include <iostream>

using Scalar = double;
using IkIdData = loik::IkIdDataTpl<Scalar>;
using IkIdDataOptimized = loik::IkIdDataTypeOptimizedTpl<Scalar>;
using MuUpdateStrat = loik::ADMMPenaltyUpdateStrat;
using FirstOrderLoik = loik::FirstOrderLoikTpl<Scalar>;
using FirstOrderLoikOptimized = loik::FirstOrderLoikOptimizedTpl<Scalar>;

IKID_DATA_TYPEDEF_TEMPLATE(IkIdDataOptimized);

boost::test_tools::predicate_result
check_scalar_abs_or_rel_equal(const Scalar a, const Scalar b, const Scalar tol = 1e-10)
{
  bool c_abs = (std::fabs(a - b) < tol);
  bool c_rel = (std::fabs(a - b) / std::fabs(a) < tol) && (std::fabs(a - b) / std::fabs(b) < tol);

  if (!c_abs && !c_rel)
  {
    boost::test_tools::predicate_result res(false);

    res.message() << "Both absolute and relative comparison failed: " << '\n'
                  << "a = " << a << ", b = " << b << ", tol = " << tol << '\n'
                  << "| a - b | = " << std::fabs(a - b) << '\n';

    return res;
  }
  return true;
}; // check_scalar_abs_or_rel_equal

template<typename T1, typename T2>
boost::test_tools::predicate_result check_eigen_dense_abs_or_rel_equal(
  const Eigen::DenseBase<T1> & a, const Eigen::DenseBase<T2> & b, const Scalar tol = 1e-10)
{
  // when a and b are not close to zero
  bool c1 = a.derived().isApprox(b.derived());

  // when a and b are close to zero, use absolute tol
  bool c2 = (a.derived() - b.derived()).template lpNorm<Eigen::Infinity>() < tol;

  if (!c1 && !c2)
  {
    boost::test_tools::predicate_result res(false);

    res.message() << "Both absolute and relative comparison failed: " << '\n'
                  << "tolerance tol : " << tol << '\n'
                  << "relative tol checking, c1 : " << c1 << '\n'
                  << "absolute tol checking, c2 : " << c2 << '\n'
                  << "| a - b |_inf : "
                  << (a.derived() - b.derived()).template lpNorm<Eigen::Infinity>() << '\n';
    return res;
  }

  return true;
}; // check_eigen_dense_abs_or_rel_equal

struct ProblemSetupFixture
{

  ProblemSetupFixture()
  {
    // solver instantiation quantities
    max_iter = 2;
    tol_abs = 1e-3;
    tol_rel = 1e-3;
    tol_primal_inf = 1e-2;
    tol_dual_inf = 1e-2;
    tol_tail_solve = 1e-1;
    rho = 1e-5;
    mu = 1e-2;
    mu_equality_scale_factor = 1e4;
    mu_update_strat = MuUpdateStrat::DEFAULT;
    num_eq_c = 1;
    eq_c_dim = 6;
    warm_start = false;
    verbose = false;
    logging = false;

    // build model and data
    // urdf_filename = EXAMPLE_ROBOT_DATA_MODEL_DIR +
    // std::string("/panda_description/urdf/panda.urdf"); 
    // urdf_filename =
    // EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/solo_description/urdf/solo.urdf");
    urdf_filename =
      EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/talos_data/robots/talos_full_v2.urdf");
    pinocchio::urdf::buildModel(urdf_filename, robot_model, false);

    // solve ik quantitites
    q = pinocchio::neutral(robot_model);
    // q << -2.79684649, -0.55090374,  0.424806  , -1.21112304, -0.89856966,
    //     0.79726132, -0.07125267,  0.13154589,  0.13171856;
    H_ref = Mat6x6::Identity();
    v_ref = Motion::Zero();
    active_task_constraint_ids.push_back(static_cast<Index>(robot_model.njoints - 1));

    const Mat6x6 Ai_identity = Mat6x6::Identity();
    Vec6 bi = Vec6::Zero();
    bi[2] = 0.5;
    Ais.push_back(Ai_identity);
    bis.push_back(bi);
    bound_magnitude = 4.0;
    lb = -bound_magnitude * DVec::Ones(robot_model.nv);
    ub = bound_magnitude * DVec::Ones(robot_model.nv);
  }

  int max_iter;
  Scalar tol_abs;
  Scalar tol_rel;
  Scalar tol_primal_inf;
  Scalar tol_dual_inf;
  Scalar tol_tail_solve;
  Scalar rho;
  Scalar mu;
  Scalar mu_equality_scale_factor;
  MuUpdateStrat mu_update_strat;
  int num_eq_c = 1;
  int eq_c_dim = 6;
  bool warm_start = false;
  bool verbose = false;
  bool logging = false;

  Model robot_model;

  std::string urdf_filename;

  DVec q;

  Mat6x6 H_ref;
  Motion v_ref;
  std::vector<Index> active_task_constraint_ids;
  PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) Ais;
  PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) bis;
  Scalar bound_magnitude;
  DVec lb;
  DVec ub;

}; // struct ProblemSetupFixture

// Define a global fixture to set the log level
struct SetLogLevel
{
  SetLogLevel()
  {
    boost::unit_test::unit_test_log.set_threshold_level(boost::unit_test::log_messages);
  }
}; // struct SetLogLevel

// Apply the fixture globally
BOOST_GLOBAL_FIXTURE(SetLogLevel);

// define boost test suite
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_FIXTURE_TEST_CASE(test_problem_setup, ProblemSetupFixture)
{
  max_iter = 200;
  int max_iter_test = 200;

  const Scalar tol_abs_test = 1e-3;
  const Scalar tol_rel_test = 1e-3;
  const Scalar tol_primal_inf_test = 1e-2;
  const Scalar tol_dual_inf_test = 1e-2;
  const Scalar rho_test = 1e-5;
  const Scalar mu_test = 1e-2;
  const Scalar mu_equality_scale_factor_test = 1e4;
  const MuUpdateStrat mu_update_strat_test = MuUpdateStrat::DEFAULT;
  int num_eq_c_test = 1;
  int eq_c_dim_test = 6;
  bool warm_start_test = false;
  bool verbose_test = false;
  bool logging_test = false;

  // empty robot model
  Model robot_model_test;

  // pinocchio::JointModelFreeFlyerTpl<Scalar> fb_joint_model;

  // build model and data
  const std::string urdf_filename_test = urdf_filename;
  pinocchio::urdf::buildModel(urdf_filename_test, robot_model_test, false);

  // solve ik quantitites
  DVec q_test = q;
  const Mat6x6 H_ref_test = Mat6x6::Identity();
  const Motion v_ref_test = Motion::Zero();
  const std::vector<Index> active_task_constraint_ids_test{
    static_cast<Index>(robot_model_test.njoints - 1)};
  PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) Ais_test;
  PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) bis_test;
  const Mat6x6 Ai_identity_test = Mat6x6::Identity();
  Vec6 bi_test = Vec6::Zero();
  bi_test[2] = 0.5;
  Ais_test.push_back(Ai_identity_test);
  bis_test.push_back(bi_test);
  const Scalar bound_magnitude_test = 4.0;
  const DVec lb_test = -bound_magnitude_test * DVec::Ones(robot_model_test.nv);
  const DVec ub_test = bound_magnitude_test * DVec::Ones(robot_model_test.nv);

  IkIdData ikid_data_test(robot_model_test, eq_c_dim_test);

  FirstOrderLoik LoikSolver_test{
    max_iter_test,        tol_abs_test,    tol_rel_test,   tol_primal_inf_test,
    tol_dual_inf_test,    rho_test,        mu_test,        mu_equality_scale_factor_test,
    mu_update_strat_test, num_eq_c_test,   eq_c_dim_test,  robot_model_test,
    ikid_data_test,       warm_start_test, tol_tail_solve, verbose_test,
    logging_test};

  LoikSolver_test.Solve(
    q_test, H_ref_test, v_ref_test, active_task_constraint_ids_test, Ais_test, bis_test, lb_test,
    ub_test);

  IkIdData ikid_data(robot_model, eq_c_dim);
  FirstOrderLoik LoikSolver{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));
  BOOST_CHECK(ikid_data_test.z.isApprox(ikid_data.z));
  BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
  BOOST_CHECK(ikid_data_test.His[1].isApprox(ikid_data.His[1]));

  BOOST_CHECK(LoikSolver_test.get_iter() == LoikSolver.get_iter());

} // test_problem_setup

BOOST_FIXTURE_TEST_CASE(test_loik_solve_split, ProblemSetupFixture)
{
  max_iter = 200;
  bound_magnitude = 5.0;
  lb = -bound_magnitude * DVec::Ones(robot_model.nv);
  ub = bound_magnitude * DVec::Ones(robot_model.nv);

  verbose = false;

  // instantiate ground truth solver
  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  // solve using full reset
  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // instantiate test solver
  IkIdData ikid_data_test(robot_model, eq_c_dim);
  FirstOrderLoik LoikSolver_test{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data_test,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  // solve with seperate Init and Solve
  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);
  LoikSolver_test.Solve();

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));
  BOOST_CHECK(ikid_data_test.z.isApprox(ikid_data.z));
  BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
  BOOST_CHECK(ikid_data_test.His[1].isApprox(ikid_data.His[1]));

  BOOST_CHECK(LoikSolver_test.get_iter() == LoikSolver.get_iter());
} // test_loik_solve_split

BOOST_FIXTURE_TEST_CASE(
  test_1st_order_loik_optimized_correctness_component_wise, ProblemSetupFixture)
{
  max_iter = 200;
  bound_magnitude = 1.0;
  lb = -bound_magnitude * DVec::Ones(robot_model.nv);
  ub = bound_magnitude * DVec::Ones(robot_model.nv);

  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  IkIdDataOptimized ikid_data_test(robot_model, num_eq_c);

  FirstOrderLoikOptimized LoikSolver_test{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data_test,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  LoikSolver.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);
  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // fwd pass 1
  LoikSolver.FwdPass1();
  LoikSolver_test.FwdPass1();

  for (const auto & idx : ikid_data_test.joint_full_range)
  {
    BOOST_CHECK(ikid_data_test.His_aba[idx].isApprox(ikid_data_test.His[idx]));

    if (idx == (Index)0)
    {
      BOOST_CHECK(ikid_data.His[0].isApprox(Mat6x6::Zero()));
      BOOST_CHECK(ikid_data_test.His[0].isApprox(Mat6x6::Identity()));
    }
    else
    {
      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx], 1e-14));
    }

    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx], 1e-14));
    BOOST_CHECK(
      (ikid_data_test.pis_aba[idx].toVector()).isApprox(ikid_data_test.pis[idx].toVector()));
  }

  for (const auto & idx : ikid_data_test.joint_range)
  {
    const JointModel & jmodel = robot_model.joints[idx];
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      jmodel.jointVelocitySelector(ikid_data_test.R), ikid_data.Ris[idx], 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      jmodel.jointVelocitySelector(ikid_data_test.r), ikid_data.ris[idx], 1e-14));
  }

  // bwd pass
  LoikSolver.BwdPass();
  // LoikSolver_test.BwdPassOptimized();
  LoikSolver_test.BwdPassOptimizedVisitor();

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx], 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx], 1e-14));
  }

  // fwd pass 2
  LoikSolver.FwdPass2();
  LoikSolver_test.FwdPass2OptimizedVisitor();

  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.nu, ikid_data.nu, 1e-14));

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.vis[idx].toVector(), ikid_data.vis[idx].toVector(), 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.fis[idx].toVector(), ikid_data.fis[idx], 1e-14));
  }

  // box proj
  LoikSolver.BoxProj();
  LoikSolver_test.BoxProj();

  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.nu, ikid_data.nu, 1e-14));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w, 1e-14));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.z, ikid_data.z, 1e-14));

  // dual update"
  LoikSolver.DualUpdate();
  LoikSolver_test.DualUpdate();
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w, 1e-14));

  Index c_vec_id = 0;
  for (const auto & c_id : active_task_constraint_ids)
  {
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.yis[c_vec_id], ikid_data.yis[c_id], 1e-14));
    c_vec_id++;
  }

  // compute residual
  LoikSolver.UpdateQPADMMSolveLoopUtility();
  LoikSolver.ComputeResiduals();
  LoikSolver_test.ComputeResiduals();

  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_primal_residual(), LoikSolver_test.get_primal_residual()));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
    LoikSolver.get_primal_residual_vec(), LoikSolver_test.get_primal_residual_vec()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_dual_residual(), LoikSolver_test.get_dual_residual()));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
    LoikSolver.get_dual_residual_vec(), LoikSolver_test.get_dual_residual_vec()));

  // check convergence
  LoikSolver.CheckConvergence();
  LoikSolver_test.CheckConvergence();

  BOOST_TEST(
    check_scalar_abs_or_rel_equal(LoikSolver.get_tol_primal(), LoikSolver_test.get_tol_primal()));
  BOOST_CHECK(LoikSolver.get_tol_primal() != 0.0);
  BOOST_CHECK(LoikSolver_test.get_tol_primal() != 0.0);

  BOOST_TEST(
    check_scalar_abs_or_rel_equal(LoikSolver.get_tol_dual(), LoikSolver_test.get_tol_dual()));
  BOOST_CHECK(LoikSolver.get_tol_dual() != 0.0);
  BOOST_CHECK(LoikSolver_test.get_tol_dual() != 0.0);

  BOOST_CHECK(LoikSolver.get_convergence_status() == LoikSolver_test.get_convergence_status());

  // check feasibility
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_y_qp_inf_norm(), LoikSolver_test.get_delta_y_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_x_qp_inf_norm(), LoikSolver_test.get_delta_x_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_z_qp_inf_norm(), LoikSolver_test.get_delta_z_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(0, 6 * (robot_model.njoints - 1)))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_fis_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(
       6 * (robot_model.njoints - 1), 6 * (robot_model.njoints - 1)))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_yis_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(2 * 6 * (robot_model.njoints - 1), robot_model.nv))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_w_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_A_qp_T_delta_y_qp_inf_norm(), LoikSolver_test.get_A_qp_T_delta_y_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_ub_qp_T_delta_y_qp_plus(), LoikSolver_test.get_ub_qp_T_delta_y_qp_plus()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_lb_qp_T_delta_y_qp_minus(), LoikSolver_test.get_lb_qp_T_delta_y_qp_minus()));

  LoikSolver.CheckFeasibility();
  LoikSolver_test.CheckFeasibility();
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_cond_1()
    == LoikSolver_test.get_primal_infeasibility_cond_1());
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_cond_2()
    == LoikSolver_test.get_primal_infeasibility_cond_2());
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_status()
    == LoikSolver_test.get_primal_infeasibility_status());

  // UpdateMu
  BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu(), 1e-14));
  LoikSolver.UpdateMu();
  LoikSolver_test.UpdateMu();
  BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu(), 1e-14));

  // Sanity Check
  for (const auto & idx : ikid_data_test.joint_range)
  {

    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx]));
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx]));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.vis[idx].toVector(), ikid_data.vis[idx].toVector()));
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.fis[idx].toVector(), ikid_data.fis[idx]));
  }

  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.nu, ikid_data.nu));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.z, ikid_data.z));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w));

  c_vec_id = 0;
  for (const auto & c_id : active_task_constraint_ids)
  {

    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.yis[c_vec_id], ikid_data.yis[c_id]));

    c_vec_id++;
  }

  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
    LoikSolver.get_primal_residual_vec(), LoikSolver_test.get_primal_residual_vec()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_primal_residual(), LoikSolver_test.get_primal_residual()));

  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_dual_residual(), LoikSolver_test.get_dual_residual()));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
    LoikSolver.get_dual_residual_vec(), LoikSolver_test.get_dual_residual_vec()));

  BOOST_TEST(
    check_scalar_abs_or_rel_equal(LoikSolver.get_tol_primal(), LoikSolver_test.get_tol_primal()));
  BOOST_CHECK(LoikSolver.get_tol_primal() != 0.0);
  BOOST_CHECK(LoikSolver_test.get_tol_primal() != 0.0);
  BOOST_TEST(
    check_scalar_abs_or_rel_equal(LoikSolver.get_tol_dual(), LoikSolver_test.get_tol_dual()));
  BOOST_CHECK(LoikSolver.get_tol_dual() != 0.0);
  BOOST_CHECK(LoikSolver_test.get_tol_dual() != 0.0);
  BOOST_CHECK(LoikSolver.get_convergence_status() == LoikSolver_test.get_convergence_status());

  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_y_qp_inf_norm(), LoikSolver_test.get_delta_y_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_x_qp_inf_norm(), LoikSolver_test.get_delta_x_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_z_qp_inf_norm(), LoikSolver_test.get_delta_z_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(0, 6 * (robot_model.njoints - 1)))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_fis_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(
       6 * (robot_model.njoints - 1), 6 * (robot_model.njoints - 1)))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_yis_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(2 * 6 * (robot_model.njoints - 1), robot_model.nv))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_w_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_A_qp_T_delta_y_qp_inf_norm(), LoikSolver_test.get_A_qp_T_delta_y_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_ub_qp_T_delta_y_qp_plus(), LoikSolver_test.get_ub_qp_T_delta_y_qp_plus()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_lb_qp_T_delta_y_qp_minus(), LoikSolver_test.get_lb_qp_T_delta_y_qp_minus()));

  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_cond_1()
    == LoikSolver_test.get_primal_infeasibility_cond_1());
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_cond_2()
    == LoikSolver_test.get_primal_infeasibility_cond_2());
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_status()
    == LoikSolver_test.get_primal_infeasibility_status());

  BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu()));

} // test_1st_order_loik_optimized_correctness_component_wise

BOOST_FIXTURE_TEST_CASE(test_1st_order_loik_optimized_correctness, ProblemSetupFixture)
{
  max_iter = 8;
  bound_magnitude = 2.0;
  lb = -bound_magnitude * DVec::Ones(robot_model.nv);
  ub = bound_magnitude * DVec::Ones(robot_model.nv);

  // instantiate ground truth solver
  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  // solve using full reset
  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // instantiate test solver
  IkIdDataOptimized ikid_data_test(robot_model, num_eq_c);
  FirstOrderLoikOptimized LoikSolver_test{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data_test,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // repeatedly call 'Solve()' and check answers against ground truth
  const int LOOP = 1;
  SMOOTH(LOOP)
  {
    LoikSolver_test.Solve();

    for (const auto & idx : ikid_data_test.joint_range)
    {
      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx]));
      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx]));
      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
        ikid_data_test.vis[idx].toVector(), ikid_data.vis[idx].toVector()));
      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.fis[idx].toVector(), ikid_data.fis[idx]));
    }

    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.nu, ikid_data.nu));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.z, ikid_data.z));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w));

    Index c_vec_id = 0;
    for (const auto & c_id : active_task_constraint_ids)
    {

      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.yis[c_vec_id], ikid_data.yis[c_id]));

      c_vec_id++;
    }

    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      LoikSolver.get_primal_residual_vec(), LoikSolver_test.get_primal_residual_vec()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_primal_residual(), LoikSolver_test.get_primal_residual()));

    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_dual_residual(), LoikSolver_test.get_dual_residual()));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      LoikSolver.get_dual_residual_vec(), LoikSolver_test.get_dual_residual_vec()));

    BOOST_TEST(
      check_scalar_abs_or_rel_equal(LoikSolver.get_tol_primal(), LoikSolver_test.get_tol_primal()));
    BOOST_CHECK(LoikSolver.get_tol_primal() != 0.0);
    BOOST_CHECK(LoikSolver_test.get_tol_primal() != 0.0);
    BOOST_TEST(
      check_scalar_abs_or_rel_equal(LoikSolver.get_tol_dual(), LoikSolver_test.get_tol_dual()));
    BOOST_CHECK(LoikSolver.get_tol_dual() != 0.0);
    BOOST_CHECK(LoikSolver_test.get_tol_dual() != 0.0);
    BOOST_CHECK(LoikSolver.get_convergence_status() == LoikSolver_test.get_convergence_status());

    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_delta_y_qp_inf_norm(), LoikSolver_test.get_delta_y_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_delta_x_qp_inf_norm(), LoikSolver_test.get_delta_x_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_delta_z_qp_inf_norm(), LoikSolver_test.get_delta_z_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      (LoikSolver.get_delta_y_qp().segment(0, 6 * (robot_model.njoints - 1)))
        .template lpNorm<Eigen::Infinity>(),
      ikid_data_test.delta_fis_inf_norm));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      (LoikSolver.get_delta_y_qp().segment(
         6 * (robot_model.njoints - 1), 6 * (robot_model.njoints - 1)))
        .template lpNorm<Eigen::Infinity>(),
      ikid_data_test.delta_yis_inf_norm));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      (LoikSolver.get_delta_y_qp().segment(2 * 6 * (robot_model.njoints - 1), robot_model.nv))
        .template lpNorm<Eigen::Infinity>(),
      ikid_data_test.delta_w_inf_norm));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_A_qp_T_delta_y_qp_inf_norm(),
      LoikSolver_test.get_A_qp_T_delta_y_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_ub_qp_T_delta_y_qp_plus(), LoikSolver_test.get_ub_qp_T_delta_y_qp_plus()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_lb_qp_T_delta_y_qp_minus(), LoikSolver_test.get_lb_qp_T_delta_y_qp_minus()));

    BOOST_CHECK(
      LoikSolver.get_primal_infeasibility_cond_1()
      == LoikSolver_test.get_primal_infeasibility_cond_1());
    BOOST_CHECK(
      LoikSolver.get_primal_infeasibility_cond_2()
      == LoikSolver_test.get_primal_infeasibility_cond_2());
    BOOST_CHECK(
      LoikSolver.get_primal_infeasibility_status()
      == LoikSolver_test.get_primal_infeasibility_status());

    BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu()));
  }

} // test_1st_order_loik_optimized_correctness

BOOST_FIXTURE_TEST_CASE(test_1st_order_loik_optimized_reset_component_wise, ProblemSetupFixture)
{
  max_iter = 100;
  bound_magnitude = 1.5;
  lb = -bound_magnitude * DVec::Ones(robot_model.nv);
  ub = bound_magnitude * DVec::Ones(robot_model.nv);

  BOOST_CHECK(warm_start == false);

  // instantiate ground truth solver
  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  // solve using full reset
  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // instantiate test solver
  IkIdDataOptimized ikid_data_test(robot_model, num_eq_c);
  FirstOrderLoikOptimized LoikSolver_test{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data_test,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // call 'Solve()' once
  LoikSolver_test.Solve();

  LoikSolver.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);
  ikid_data.UpdatePrev();

  // reset IkIdData recursion quantites
  ikid_data_test.ResetRecursion();

  // wipe solver quantities'
  LoikSolver_test.ResetSolver();

  // solver main loop starts here
  ikid_data_test.UpdatePrev();

  ikid_data_test.ResetInfNorms();

  // fwd pass 1
  LoikSolver.FwdPass1();
  LoikSolver_test.FwdPass1();

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_CHECK(ikid_data_test.His_aba[idx].isApprox(ikid_data_test.His[idx]));
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx], 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx], 1e-14));
    BOOST_CHECK(
      (ikid_data_test.pis_aba[idx].toVector()).isApprox(ikid_data_test.pis[idx].toVector()));
  }

  for (const auto & idx : ikid_data_test.joint_range)
  {
    const JointModel & jmodel = robot_model.joints[idx];
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      jmodel.jointVelocitySelector(ikid_data_test.R), ikid_data.Ris[idx], 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      jmodel.jointVelocitySelector(ikid_data_test.r), ikid_data.ris[idx], 1e-14));
  }

  // bwd pass 1
  LoikSolver.BwdPass();
  LoikSolver_test.BwdPassOptimizedVisitor();

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx], 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx], 1e-14));
  }

  // fwd pass 2
  LoikSolver.FwdPass2();
  LoikSolver_test.FwdPass2OptimizedVisitor();

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.vis[idx].toVector(), ikid_data.vis[idx].toVector(), 1e-14));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      ikid_data_test.fis[idx].toVector(), ikid_data.fis[idx], 1e-14));
  }

  // box proj
  LoikSolver.BoxProj();
  LoikSolver_test.BoxProj();

  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.nu, ikid_data.nu, 1e-14));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w, 1e-14));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.z, ikid_data.z, 1e-14));

  // dual update"
  LoikSolver.DualUpdate();
  LoikSolver_test.DualUpdate();
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w, 1e-14));

  Index c_vec_id = 0;
  for (const auto & c_id : active_task_constraint_ids)
  {
    BOOST_TEST(
      check_eigen_dense_abs_or_rel_equal(ikid_data_test.yis[c_vec_id], ikid_data.yis[c_id], 1e-14));

    c_vec_id++;
  }

  // compute residual
  LoikSolver.UpdateQPADMMSolveLoopUtility();
  LoikSolver.ComputeResiduals();
  LoikSolver_test.ComputeResiduals();

  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_primal_residual(), LoikSolver_test.get_primal_residual()));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
    LoikSolver.get_primal_residual_vec(), LoikSolver_test.get_primal_residual_vec()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_dual_residual(), LoikSolver_test.get_dual_residual()));
  BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
    LoikSolver.get_dual_residual_vec(), LoikSolver_test.get_dual_residual_vec()));

  // check convergence
  LoikSolver.CheckConvergence();
  LoikSolver_test.CheckConvergence();

  BOOST_TEST(
    check_scalar_abs_or_rel_equal(LoikSolver.get_tol_primal(), LoikSolver_test.get_tol_primal()));
  BOOST_CHECK(LoikSolver.get_tol_primal() != 0.0);
  BOOST_CHECK(LoikSolver_test.get_tol_primal() != 0.0);

  BOOST_TEST(
    check_scalar_abs_or_rel_equal(LoikSolver.get_tol_dual(), LoikSolver_test.get_tol_dual()));
  BOOST_CHECK(LoikSolver.get_tol_dual() != 0.0);
  BOOST_CHECK(LoikSolver_test.get_tol_dual() != 0.0);

  BOOST_CHECK(LoikSolver.get_convergence_status() == LoikSolver_test.get_convergence_status());

  // check feasibility
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_delta_y_qp_inf_norm(), LoikSolver_test.get_delta_y_qp_inf_norm()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(0, 6 * (robot_model.njoints - 1)))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_fis_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(
       6 * (robot_model.njoints - 1), 6 * (robot_model.njoints - 1)))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_yis_inf_norm));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    (LoikSolver.get_delta_y_qp().segment(2 * 6 * (robot_model.njoints - 1), robot_model.nv))
      .template lpNorm<Eigen::Infinity>(),
    ikid_data_test.delta_w_inf_norm));

  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_A_qp_T_delta_y_qp_inf_norm(), LoikSolver_test.get_A_qp_T_delta_y_qp_inf_norm()));

  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_ub_qp_T_delta_y_qp_plus(), LoikSolver_test.get_ub_qp_T_delta_y_qp_plus()));
  BOOST_TEST(check_scalar_abs_or_rel_equal(
    LoikSolver.get_lb_qp_T_delta_y_qp_minus(), LoikSolver_test.get_lb_qp_T_delta_y_qp_minus()));

  LoikSolver.CheckFeasibility();
  LoikSolver_test.CheckFeasibility();
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_cond_1()
    == LoikSolver_test.get_primal_infeasibility_cond_1());
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_cond_2()
    == LoikSolver_test.get_primal_infeasibility_cond_2());
  BOOST_CHECK(
    LoikSolver.get_primal_infeasibility_status()
    == LoikSolver_test.get_primal_infeasibility_status());

  // UpdateMu
  BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu(), 1e-14));
  LoikSolver.UpdateMu();
  LoikSolver_test.UpdateMu();
  BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu(), 1e-14));

  // Check iteration count
  BOOST_CHECK(LoikSolver.get_iter() == LoikSolver_test.get_iter());

} // test_1st_order_loik_optimized_reset_component_wise

BOOST_FIXTURE_TEST_CASE(test_1st_order_loik_optimized_reset, ProblemSetupFixture)
{
  max_iter = 100;
  bound_magnitude = 2.0;
  lb = -bound_magnitude * DVec::Ones(robot_model.nv);
  ub = bound_magnitude * DVec::Ones(robot_model.nv);

  // instantiate ground truth solver
  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  // solve using full reset
  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // instantiate test solver
  IkIdDataOptimized ikid_data_test(robot_model, num_eq_c);
  FirstOrderLoikOptimized LoikSolver_test{
    max_iter,
    tol_abs,
    tol_rel,
    tol_primal_inf,
    tol_dual_inf,
    rho,
    mu,
    mu_equality_scale_factor,
    mu_update_strat,
    num_eq_c,
    eq_c_dim,
    robot_model,
    ikid_data_test,
    warm_start,
    tol_tail_solve,
    verbose,
    logging};

  // LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // repeatedly call 'Solve()' and check answers against ground truth
  const int LOOP = 5;
  SMOOTH(LOOP)
  {
    // LoikSolver_test.Solve();

    LoikSolver_test.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);
    // LoikSolver_test.Solve(q, active_task_constraint_ids[0], Ais[0], bis[0]);

    for (const auto & idx : ikid_data_test.joint_range)
    {

      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.His[idx], ikid_data.His[idx]));
      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.pis[idx].toVector(), ikid_data.pis[idx]));
      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
        ikid_data_test.vis[idx].toVector(), ikid_data.vis[idx].toVector()));
      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.fis[idx].toVector(), ikid_data.fis[idx]));
    }

    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.nu, ikid_data.nu));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.z, ikid_data.z));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data_test.w, ikid_data.w));

    Index c_vec_id = 0;
    for (const auto & c_id : active_task_constraint_ids)
    {

      BOOST_TEST(
        check_eigen_dense_abs_or_rel_equal(ikid_data_test.yis[c_vec_id], ikid_data.yis[c_id]));

      c_vec_id++;
    }

    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      LoikSolver.get_primal_residual_vec(), LoikSolver_test.get_primal_residual_vec()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_primal_residual(), LoikSolver_test.get_primal_residual()));

    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_dual_residual(), LoikSolver_test.get_dual_residual()));
    BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
      LoikSolver.get_dual_residual_vec(), LoikSolver_test.get_dual_residual_vec()));

    BOOST_TEST(
      check_scalar_abs_or_rel_equal(LoikSolver.get_tol_primal(), LoikSolver_test.get_tol_primal()));
    BOOST_CHECK(LoikSolver.get_tol_primal() != 0.0);
    BOOST_CHECK(LoikSolver_test.get_tol_primal() != 0.0);
    BOOST_TEST(
      check_scalar_abs_or_rel_equal(LoikSolver.get_tol_dual(), LoikSolver_test.get_tol_dual()));
    BOOST_CHECK(LoikSolver.get_tol_dual() != 0.0);
    BOOST_CHECK(LoikSolver_test.get_tol_dual() != 0.0);
    BOOST_CHECK(LoikSolver.get_convergence_status() == LoikSolver_test.get_convergence_status());

    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_delta_y_qp_inf_norm(), LoikSolver_test.get_delta_y_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_delta_x_qp_inf_norm(), LoikSolver_test.get_delta_x_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_delta_z_qp_inf_norm(), LoikSolver_test.get_delta_z_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      (LoikSolver.get_delta_y_qp().segment(0, 6 * (robot_model.njoints - 1)))
        .template lpNorm<Eigen::Infinity>(),
      ikid_data_test.delta_fis_inf_norm));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      (LoikSolver.get_delta_y_qp().segment(
         6 * (robot_model.njoints - 1), 6 * (robot_model.njoints - 1)))
        .template lpNorm<Eigen::Infinity>(),
      ikid_data_test.delta_yis_inf_norm));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      (LoikSolver.get_delta_y_qp().segment(2 * 6 * (robot_model.njoints - 1), robot_model.nv))
        .template lpNorm<Eigen::Infinity>(),
      ikid_data_test.delta_w_inf_norm));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_A_qp_T_delta_y_qp_inf_norm(),
      LoikSolver_test.get_A_qp_T_delta_y_qp_inf_norm()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_ub_qp_T_delta_y_qp_plus(), LoikSolver_test.get_ub_qp_T_delta_y_qp_plus()));
    BOOST_TEST(check_scalar_abs_or_rel_equal(
      LoikSolver.get_lb_qp_T_delta_y_qp_minus(), LoikSolver_test.get_lb_qp_T_delta_y_qp_minus()));

    BOOST_CHECK(
      LoikSolver.get_primal_infeasibility_cond_1()
      == LoikSolver_test.get_primal_infeasibility_cond_1());
    BOOST_CHECK(
      LoikSolver.get_primal_infeasibility_cond_2()
      == LoikSolver_test.get_primal_infeasibility_cond_2());
    BOOST_CHECK(
      LoikSolver.get_primal_infeasibility_status()
      == LoikSolver_test.get_primal_infeasibility_status());

    BOOST_TEST(check_scalar_abs_or_rel_equal(LoikSolver.get_mu(), LoikSolver_test.get_mu()));

    BOOST_CHECK(LoikSolver.get_iter() == LoikSolver_test.get_iter());
  }

} // test_1st_order_loik_optimized_reset


BOOST_AUTO_TEST_SUITE_END()
