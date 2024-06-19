//
// Copyright (c) 2024 INRIA
//

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
using Model = pinocchio::ModelTpl<Scalar>;
using IkIdData = loik::IkIdDataTpl<Scalar>;
using IkIdDataOptimized = loik::IkIdDataTypeOptimizedTpl<Scalar>;
using JointModel = IkIdDataOptimized::JointModel;
using Inertia = typename IkIdData::Inertia;
using Motion = typename IkIdData::Motion;
using Force = typename IkIdData::Force;
using DMat = typename IkIdData::DMat;
using Mat6x6 = typename IkIdData::Mat6x6;
using DVec = typename IkIdData::DVec;
using Vec6 = typename IkIdData::Vec6;
using Index = typename IkIdData::Index;
using MuUpdateStrat = loik::ADMMPenaltyUpdateStrat;
using FirstOrderLoik = loik::FirstOrderLoikTpl<Scalar>;
using FirstOrderLoikOptimized = loik::FirstOrderLoikOptimizedTpl<Scalar>;

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
    rho = 1e-5;
    mu = 1e-2;
    mu_equality_scale_factor = 1e4;
    mu_update_strat = MuUpdateStrat::DEFAULT;
    num_eq_c = 1;
    eq_c_dim = 6;
    warm_start = false;
    verbose = false;
    logging = false;

    // pinocchio::JointModelFreeFlyerTpl<Scalar> fb_joint_model;

    // build model and data
    urdf_filename =
      EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/panda_description/urdf/panda.urdf");
    pinocchio::urdf::buildModel(urdf_filename, robot_model, false);

    // solve ik quantitites
    q = pinocchio::neutral(robot_model);
    q << -2.79684649, -0.55090374, 0.424806, -1.21112304, -0.89856966, 0.79726132, -0.07125267,
      0.13154589, 0.13171856;
    H_ref = Mat6x6::Identity();
    H_ref_inertia = Inertia{H_ref};
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
  Inertia H_ref_inertia;
  Motion v_ref;
  std::vector<Index> active_task_constraint_ids;
  PINOCCHIO_ALIGNED_STD_VECTOR(Mat6x6) Ais;
  PINOCCHIO_ALIGNED_STD_VECTOR(Vec6) bis;
  Scalar bound_magnitude;
  DVec lb;
  DVec ub;
};

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
  const std::string urdf_filename_test =
    EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/panda_description/urdf/panda.urdf");
  pinocchio::urdf::buildModel(urdf_filename_test, robot_model_test, false);

  // solve ik quantitites
  DVec q_test = pinocchio::neutral(robot_model);
  q_test << -2.79684649, -0.55090374, 0.424806, -1.21112304, -0.89856966, 0.79726132, -0.07125267,
    0.13154589, 0.13171856;
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
    max_iter_test,        tol_abs_test,    tol_rel_test,  tol_primal_inf_test,
    tol_dual_inf_test,    rho_test,        mu_test,       mu_equality_scale_factor_test,
    mu_update_strat_test, num_eq_c_test,   eq_c_dim_test, robot_model_test,
    ikid_data_test,       warm_start_test, verbose_test,  logging_test};

  LoikSolver_test.Solve(
    q_test, H_ref_test, v_ref_test, active_task_constraint_ids_test, Ais_test, bis_test, lb_test,
    ub_test);

  IkIdData ikid_data(robot_model, eq_c_dim);
  FirstOrderLoik LoikSolver{max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
                            tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
                            mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
                            ikid_data,       warm_start, verbose,  logging};

  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));
  BOOST_CHECK(ikid_data_test.z.isApprox(ikid_data.z));
  BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
  BOOST_CHECK(ikid_data_test.His[1].isApprox(ikid_data.His[1]));

  BOOST_CHECK(LoikSolver_test.get_iter() == LoikSolver.get_iter());
}

BOOST_FIXTURE_TEST_CASE(test_loik_solve_split, ProblemSetupFixture)
{
  max_iter = 200;
  verbose = true;

  // instantiate ground truth solver
  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
                            tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
                            mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
                            ikid_data,       warm_start, verbose,  logging};

  // solve using full reset
  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // instantiate test solver
  IkIdData ikid_data_test(robot_model, eq_c_dim);
  FirstOrderLoik LoikSolver_test{max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
                                 tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
                                 mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
                                 ikid_data_test,  warm_start, verbose,  logging};

  // solve with seperate Init and Solve
  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);
  LoikSolver_test.Solve();

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));
  BOOST_CHECK(ikid_data_test.z.isApprox(ikid_data.z));
  BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
  BOOST_CHECK(ikid_data_test.His[1].isApprox(ikid_data.His[1]));

  BOOST_CHECK(LoikSolver_test.get_iter() == LoikSolver.get_iter());
}

BOOST_FIXTURE_TEST_CASE(test_loik_reset, ProblemSetupFixture)
{
  max_iter = 4;
  // verbose = false;

  // instantiate ground truth solver
  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
                            tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
                            mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
                            ikid_data,       warm_start, verbose,  logging};

  // solve using full reset
  LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // instantiate test solver
  IkIdDataOptimized ikid_data_test(robot_model, eq_c_dim);
  FirstOrderLoikOptimized LoikSolver_test{
    max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
    tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
    mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
    ikid_data_test,  warm_start, verbose,  logging};

  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // repeatedly call 'Solve()' and check answers against ground truth
  const int LOOP = 4;
  SMOOTH(LOOP)
  {
    LoikSolver_test.Solve();

    BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));
    BOOST_CHECK(ikid_data_test.z.isApprox(ikid_data.z));
    BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
    BOOST_CHECK(ikid_data_test.His_aba[1].isApprox(ikid_data.His_aba[1]));

    BOOST_CHECK(LoikSolver_test.get_iter() == LoikSolver.get_iter());
  }
}

BOOST_FIXTURE_TEST_CASE(test_1st_order_loik_timing, ProblemSetupFixture)
{
  max_iter = 2;

  IkIdDataOptimized ikid_data(robot_model, eq_c_dim);

  FirstOrderLoikOptimized LoikSolver{
    max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
    tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
    mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
    ikid_data,       warm_start, verbose,  logging};

  PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
  const int NBT = 100000;
// const int NBT = 1;
#else
  const int NBT = 100000;
  std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

  // timer.tic();
  // SMOOTH(NBT)
  // {
  //     LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis,
  //     lb, ub);
  // }
  // std::cout << "LOIK = \t\t\t\t"; timer.toc(std::cout,NBT);

  LoikSolver.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  timer.tic();
  SMOOTH(NBT)
  {
    // LoikSolver.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais,
    // bis, lb, ub);
    LoikSolver.Solve();
  }
  std::cout << "LOIK = \t\t\t\t";
  timer.toc(std::cout, NBT);

  // std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++" <<
  // std::endl; LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids,
  // Ais, bis, lb, ub); std::cout <<
  // "--------------------------------------------------" << std::endl;

  BOOST_CHECK(0 == 0);
}

BOOST_FIXTURE_TEST_CASE(test_1st_order_loik_optimized_correctness, ProblemSetupFixture)
{

  IkIdData ikid_data(robot_model, eq_c_dim);

  FirstOrderLoik LoikSolver{max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
                            tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
                            mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
                            ikid_data,       warm_start, verbose,  logging};

  // LoikSolver.Solve(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb,
  // ub);

  IkIdDataOptimized ikid_data_test(robot_model, eq_c_dim);

  FirstOrderLoikOptimized LoikSolver_test{
    max_iter,        tol_abs,    tol_rel,  tol_primal_inf,
    tol_dual_inf,    rho,        mu,       mu_equality_scale_factor,
    mu_update_strat, num_eq_c,   eq_c_dim, robot_model,
    ikid_data_test,  warm_start, verbose,  logging};

  // PinocchioTicToc timer_test(PinocchioTicToc::US);

  LoikSolver.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);
  LoikSolver_test.SolveInit(q, H_ref, v_ref, active_task_constraint_ids, Ais, bis, lb, ub);

  // fwd pass 1
  LoikSolver.FwdPass1();
  LoikSolver_test.FwdPass1();

  for (const auto & idx : ikid_data_test.joint_full_range)
  {
    BOOST_CHECK(ikid_data_test.His_aba[idx].isApprox(ikid_data_test.His[idx]));
    BOOST_CHECK(ikid_data_test.His[idx].isApprox(ikid_data.His[idx]));
    BOOST_CHECK((ikid_data_test.pis[idx].toVector()).isApprox(ikid_data.pis[idx]));
    BOOST_CHECK(
      (ikid_data_test.pis_aba[idx].toVector()).isApprox(ikid_data_test.pis[idx].toVector()));
  }

  // test case for R against Ris, and r against ris
  //  std::cout << "ikid_data_test.R: " << std::endl;
  //  std::cout << ikid_data_test.R << std::endl;
  //  std::cout << "ikid_data.Ris: " << std::endl;
  //  for (const auto& idx : ikid_data_test.joint_full_range) {
  //      std::cout << ikid_data.Ris[idx] << std::endl;
  //  }
  for (const auto & idx : ikid_data_test.joint_range)
  {
    // std::cout << "idx: " << idx << std::endl;
    const JointModel & jmodel = robot_model.joints[idx];
    // std::cout << "jmodel.idx_v: " << jmodel.idx_v() << std::endl;
    // std::cout << "jmodel.nv: " << jmodel.nv() << std::endl;
    // std::cout << "ikid_data_test.R segment: " <<
    // ikid_data_test.R.segment(jmodel.idx_v(), jmodel.nv()) << std::endl;
    BOOST_CHECK((jmodel.jointVelocitySelector(ikid_data_test.R)).isApprox(ikid_data.Ris[idx]));
    // std::cout << "ikid_data_test.R: " <<
    // jmodel.jointVelocitySelector(ikid_data_test.R) << std::endl; std::cout <<
    // "ikid_data.Ris: " << ikid_data.Ris[idx] << std::endl;
    BOOST_CHECK((jmodel.jointVelocitySelector(ikid_data_test.r)).isApprox(ikid_data.ris[idx]));
  }

  // bwd pass
  LoikSolver.BwdPass();
  // LoikSolver_test.BwdPassOptimized();
  LoikSolver_test.BwdPassOptimizedVisitor();

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_CHECK(ikid_data_test.His[idx].isApprox(ikid_data.His[idx]));
    // std::cout << "ikid_data_test.His: " << ikid_data_test.His[idx] <<
    // std::endl; std::cout << "ikid_data.His: " << ikid_data.His[idx] <<
    // std::endl;
    BOOST_CHECK((ikid_data_test.pis[idx].toVector()).isApprox(ikid_data.pis[idx]));
    // std::cout << "ikid_data_test.pis: " << ikid_data_test.pis[idx].toVector()
    // << std::endl; std::cout << "ikid_data.pis: " << ikid_data.pis[idx] <<
    // std::endl;
  }

  // BOOST_CHECK(ikid_data_test.His[test_id].isApprox(ikid_data.His[test_id]));
  // //
  // BOOST_CHECK(ikid_data_test.His_aba[test_id].isApprox(ikid_data_test.His[test_id]));
  // std::cout << "ikid_data.His " << std::endl;
  // std::cout << ikid_data.His[test_id] << std::endl;
  // std::cout << "ikid_data_test.His: " << std::endl;
  // std::cout << ikid_data_test.His[test_id] << std::endl;
  // std::cout << "ikid_data_test.His_aba: " << std::endl;
  // std::cout << ikid_data_test.His_aba[test_id] << std::endl;
  // std::cout << "His diff inf norm: " << std::endl;
  // std::cout << (ikid_data_test.His[test_id] -
  // ikid_data.His[test_id]).template lpNorm<Eigen::Infinity>() << std::endl;
  // std::cout << "His and His_aba diff inf norm: " << std::endl;
  // std::cout << (ikid_data_test.His_aba[test_id] -
  // ikid_data_test.His[test_id]).template lpNorm<Eigen::Infinity>() <<
  // std::endl;

  // BOOST_CHECK((ikid_data_test.pis[test_id].toVector()).isApprox(ikid_data.pis[test_id]));
  // std::cout << "ikid_data.pis: " << std::endl;
  // std::cout << ikid_data.pis[test_id] << std::endl;
  // std::cout << "ikid_data_test.pis: " << std::endl;
  // std::cout << ikid_data_test.pis[test_id].toVector() << std::endl;
  // std::cout << "pis diff inf norm: " << std::endl;
  // std::cout << (ikid_data_test.pis[test_id].toVector() -
  // ikid_data.pis[test_id]).template lpNorm<Eigen::Infinity>() << std::endl;

  // fwd pass 2
  LoikSolver.FwdPass2();
  // LoikSolver_test.FwdPass2Optimized();
  LoikSolver_test.FwdPass2OptimizedVisitor();

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));

  for (const auto & idx : ikid_data_test.joint_range)
  {
    BOOST_CHECK(ikid_data_test.vis[idx].isApprox(ikid_data.vis[idx]));
    BOOST_CHECK((ikid_data_test.fis[idx].toVector()).isApprox(ikid_data.fis[idx]));
  }
  BOOST_CHECK(ikid_data_test.vis[1].isApprox(ikid_data.vis[1]));
  BOOST_CHECK(ikid_data_test.vis[4].isApprox(ikid_data.vis[4]));
  BOOST_CHECK(ikid_data_test.vis[7].isApprox(ikid_data.vis[7]));
  BOOST_CHECK((ikid_data_test.fis[1].toVector()).isApprox(ikid_data.fis[1]));
  BOOST_CHECK((ikid_data_test.fis[4].toVector()).isApprox(ikid_data.fis[4]));
  BOOST_CHECK((ikid_data_test.fis[7].toVector()).isApprox(ikid_data.fis[7]));

  // box proj
  LoikSolver.BoxProj();
  LoikSolver_test.BoxProj();

  BOOST_CHECK(ikid_data_test.nu.isApprox(ikid_data.nu));
  BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
  BOOST_CHECK(ikid_data_test.z.isApprox(ikid_data.z));

  // dual update
  LoikSolver.DualUpdate();
  LoikSolver_test.DualUpdate();
  BOOST_CHECK(ikid_data_test.w.isApprox(ikid_data.w));
  for (const auto & idx : ikid_data_test.joint_range)
  {
    // std::cout << "ikid_data_test.yis: " << ikid_data_test.yis[idx] <<
    // std::endl; std::cout << "ikid_data.yis: " << ikid_data.yis[idx] <<
    // std::endl;
    BOOST_CHECK(ikid_data_test.yis[idx].isApprox(ikid_data.yis[idx]));
  }

  // compute residual
  LoikSolver.UpdateQPADMMSolveLoopUtility();
  LoikSolver.ComputeResiduals();
  LoikSolver_test.ComputeResiduals();

  BOOST_CHECK(LoikSolver.get_primal_residual() == LoikSolver_test.get_primal_residual());
  BOOST_CHECK(
    LoikSolver.get_primal_residual_vec().isApprox(LoikSolver_test.get_primal_residual_vec()));
  BOOST_CHECK(LoikSolver.get_dual_residual() == LoikSolver_test.get_dual_residual());
  BOOST_CHECK(LoikSolver.get_dual_residual_vec().isApprox(LoikSolver_test.get_dual_residual_vec()));
  // std::cout << "primal residual from LoikSolver: " << LoikSolver.get_primal_residual() <<
  // std::endl; std::cout << "primal residual from LoikSolver_test: " <<
  // LoikSolver_test.get_primal_residual()
  //           << std::endl;
  // std::cout << "dual residual from LoikSolver: " << LoikSolver.get_dual_residual() << std::endl;
  // std::cout << "dual residual from LoikSolver_test: " << LoikSolver_test.get_dual_residual()
  //           << std::endl;
  // std::cout << "dual residual v from LoikSolver: " << LoikSolver.get_dual_residual_v() <<
  // std::endl; std::cout << "dual residual v from LoikSolver_test: " <<
  // LoikSolver_test.get_dual_residual_v()
  //           << std::endl;
  // std::cout << "dual residual nu from LoikSolver: " << LoikSolver.get_dual_residual_nu()
  //           << std::endl;
  // std::cout << "dual residual nu from LoikSolver_test: " <<
  // LoikSolver_test.get_dual_residual_nu()
  //           << std::endl;
  // std::cout << "dual residual vec from LoikSolver: "
  //           << LoikSolver.get_dual_residual_vec().transpose() << std::endl;
  // std::cout << "dual residual vec from LoikSolver_test: "
  //           << LoikSolver_test.get_dual_residual_vec().transpose() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_pinocchio_types)
{
  pinocchio::SE3 liMi = pinocchio::SE3::Random();
  pinocchio::Force f = pinocchio::Force::Random();

  BOOST_CHECK(
    (liMi.actInv(f).toVector()).isApprox(liMi.toActionMatrix().transpose() * f.toVector()));

  Mat6x6 R6 = Vec6::Random().asDiagonal();
  Mat6x6 RtR = R6.transpose() + R6;

  // pinocchio::check_expression_if_real()

  // std::cout << "inf norm 'RtR - RtR.transpose()' = " << (RtR -
  // RtR.transpose()). template lpNorm<Eigen::Infinity>() << std::endl;
  // BOOST_CHECK((RtR - RtR.transpose()).isApprox(Mat6x6::Zero()));

  // Eigen::SelfAdjointEigenSolver<Mat6x6> eigensolver(RtR);
  // if (eigensolver.info() != Eigen::Success) {
  //     throw std::runtime_error("Eigen decomposition failed.");
  // }

  // Vec6 eigenvalues = eigensolver.eigenvalues();
  // Mat6x6 eigenvectors = eigensolver.eigenvectors();

  // // Ensure all eigenvalues are positive
  // for (int i = 0; i < eigenvalues.size(); ++i) {
  //     if (eigenvalues(i) < 1e-10) {
  //         eigenvalues(i) = 1e-8; // Make small or negative eigenvalues
  //         positive
  //     }
  // }

  // Mat6x6 D = eigenvalues.asDiagonal();
  // Mat6x6 RtR_pd = eigenvectors * D * eigenvectors.transpose();

  Vec6 v6r = Vec6::Random();

  Inertia RtR_inertia = Inertia{RtR};
  Motion v6r_motion = Motion{v6r};

  BOOST_CHECK(RtR.isApprox(RtR_inertia.matrix()));

  Force f_test = RtR_inertia * v6r_motion;
  Vec6 f_test_vec = RtR_inertia.matrix() * v6r_motion.toVector();

  BOOST_CHECK(f_test.toVector().isApprox(f_test_vec));
}

BOOST_AUTO_TEST_SUITE_END()
