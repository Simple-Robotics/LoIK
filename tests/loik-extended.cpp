//
// Copyright (c) 2024 INRIA
//

#include "loik/fwd.hpp"
#include "loik/loik-loid-data-optimized.hpp"
#include "loik/loik-loid.hpp"
#include "problems/problem-utils.hpp"

#include <pinocchio/parsers/urdf.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

#include <iostream>

using Scalar = double;
using Model = typename loik::IkIdDataTypeOptimizedTpl<Scalar>::Model;
using IkIdDataOptimized = loik::IkIdDataTypeOptimizedTpl<Scalar>;
using MuUpdateStrat = loik::ADMMPenaltyUpdateStrat;
using FirstOrderLoikOptimized = loik::FirstOrderLoikOptimizedTpl<Scalar>;

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

// define boost test suite
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_panda)
{

	loik::SequenceDiffIKProblems<Scalar> problems;
	Model robot_model;

	const std::string problem_data_file_name = LOIK_PROBLEM_DATA_DIR + std::string("/panda_problems.json");
	const std::string urdf_filename = 
			EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/panda_description/urdf/panda.urdf"); 
	pinocchio::urdf::buildModel(urdf_filename, robot_model, false);

	problems.LoadProblemsFromJson(problem_data_file_name);

	/// sanity check, make sure `nq` are the same between robot_models used by c++ and python
	if (robot_model.nq != static_cast<int>(problems.problem_sequence[0].q.size())) {
		std::cerr << "nq are not the same !!!" << std::endl;
		std::cerr << "robot_model.nq: " << robot_model.nq << std::endl;
		std::cerr << "python `q` dimension: " << problems.problem_sequence[0].q.size() << std::endl;
		BOOST_CHECK(1 == 0);
	}

	for (const auto& problem : problems.problem_sequence) {
		IkIdDataOptimized ikid_data(robot_model, problem.num_eq_c);

		MuUpdateStrat mu_update_strat = MuUpdateStrat::DEFAULT;

		FirstOrderLoikOptimized LoikSolver{
																				problem.max_iter,
																				problem.tol_abs,
																				problem.tol_rel,
																				problem.tol_primal_inf,
																				problem.tol_dual_inf,
																				problem.rho,
																				problem.mu,
																				problem.mu_equality_scale_factor,
																				mu_update_strat,
																				problem.num_eq_c,
																				problem.eq_c_dim,
																				robot_model,
																				ikid_data,
																				problem.warm_start,
																				problem.tol_tail_solve,
																				problem.verbose,
																				problem.logging
																			};

		LoikSolver.Solve(
											problem.q, 
											problem.H_refs[0], 
											problem.v_refs[0], 
											problem.active_task_constraint_ids, 
											problem.Ais, 
											problem.bis, 
											problem.lb, 
											problem.ub
										);

		for (const auto & idx : ikid_data.joint_range)
    {
			if ((ikid_data.vis[idx].toVector() - problem.vis[idx].toVector()).template lpNorm<Eigen::Infinity>() >= 1e-3) {
				std::cout << problem.name << std::endl;
				std::cout << "idx: " << idx << std::endl;
				std::cout << "vis[idx] inf norm: " << (ikid_data.vis[idx].toVector() - problem.vis[idx].toVector()).template lpNorm<Eigen::Infinity>() << std::endl;
			}
      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
        ikid_data.vis[idx].toVector(), problem.vis[idx].toVector(), 
				1e-3));
    }

		BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data.z, problem.z, 1e-3));


	}

} // test_panda


BOOST_AUTO_TEST_CASE(test_talos)
{

	loik::SequenceDiffIKProblems<Scalar> problems;
	Model robot_model;

	const std::string problem_data_file_name = LOIK_PROBLEM_DATA_DIR + std::string("/talos_problems.json");
	const std::string urdf_filename =
	  EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/talos_data/robots/talos_full_v2.urdf");
	pinocchio::JointModelFreeFlyerTpl<Scalar> fb_joint_model;
	pinocchio::urdf::buildModel(urdf_filename, fb_joint_model, robot_model, false);

	problems.LoadProblemsFromJson(problem_data_file_name);

	/// sanity check, make sure `nq` are the same between robot_models used by c++ and python
	if (robot_model.nq != static_cast<int>(problems.problem_sequence[0].q.size())) {
		std::cerr << "nq are not the same !!!" << std::endl;
		std::cerr << "robot_model.nq: " << robot_model.nq << std::endl;
		std::cerr << "python `q` dimension: " << problems.problem_sequence[0].q.size() << std::endl;
		BOOST_CHECK(1 == 0);
	}

	for (const auto& problem : problems.problem_sequence) {
		IkIdDataOptimized ikid_data(robot_model, problem.num_eq_c);

		MuUpdateStrat mu_update_strat = MuUpdateStrat::DEFAULT;

		FirstOrderLoikOptimized LoikSolver{
																				problem.max_iter,
																				problem.tol_abs,
																				problem.tol_rel,
																				problem.tol_primal_inf,
																				problem.tol_dual_inf,
																				problem.rho,
																				problem.mu,
																				problem.mu_equality_scale_factor,
																				mu_update_strat,
																				problem.num_eq_c,
																				problem.eq_c_dim,
																				robot_model,
																				ikid_data,
																				problem.warm_start,
																				problem.tol_tail_solve,
																				problem.verbose,
																				problem.logging
																			};

		LoikSolver.Solve(
											problem.q, 
											problem.H_refs[0], 
											problem.v_refs[0], 
											problem.active_task_constraint_ids, 
											problem.Ais, 
											problem.bis, 
											problem.lb, 
											problem.ub
										);

		for (const auto & idx : ikid_data.joint_range)
    {
      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
        ikid_data.vis[idx].toVector(), problem.vis[idx].toVector(), 
				1e-4));
    }

		BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data.z, problem.z, 1e-4));


	}

} // test_talos


/// TODO: this test currently fails due to wrong urdf for romeo being loaded with latest version of example-robot-data
BOOST_AUTO_TEST_CASE(test_romeo)
{

	loik::SequenceDiffIKProblems<Scalar> problems;
	Model robot_model;

	const std::string problem_data_file_name = LOIK_PROBLEM_DATA_DIR + std::string("/romeo_problems.json");
	const std::string urdf_filename =
	  EXAMPLE_ROBOT_DATA_MODEL_DIR + std::string("/romeo_description/urdf/romeo.urdf");
	pinocchio::JointModelFreeFlyerTpl<Scalar> fb_joint_model;
	pinocchio::urdf::buildModel(urdf_filename, fb_joint_model, robot_model, false);

	problems.LoadProblemsFromJson(problem_data_file_name);

	/// sanity check, make sure `nq` are the same between robot_models used by c++ and python
	if (robot_model.nq != static_cast<int>(problems.problem_sequence[0].q.size())) {
		std::cerr << "nq are not the same !!!" << std::endl;
		std::cerr << "robot_model.nq: " << robot_model.nq << std::endl;
		std::cerr << "python `q` dimension: " << problems.problem_sequence[0].q.size() << std::endl;
		BOOST_CHECK(1 == 0);
	}

	for (const auto& problem : problems.problem_sequence) {
		IkIdDataOptimized ikid_data(robot_model, problem.num_eq_c);

		MuUpdateStrat mu_update_strat = MuUpdateStrat::DEFAULT;

		FirstOrderLoikOptimized LoikSolver{
																				problem.max_iter,
																				problem.tol_abs,
																				problem.tol_rel,
																				problem.tol_primal_inf,
																				problem.tol_dual_inf,
																				problem.rho,
																				problem.mu,
																				problem.mu_equality_scale_factor,
																				mu_update_strat,
																				problem.num_eq_c,
																				problem.eq_c_dim,
																				robot_model,
																				ikid_data,
																				problem.warm_start,
																				problem.tol_tail_solve,
																				problem.verbose,
																				problem.logging
																			};

		LoikSolver.Solve(
											problem.q, 
											problem.H_refs[0], 
											problem.v_refs[0], 
											problem.active_task_constraint_ids, 
											problem.Ais, 
											problem.bis, 
											problem.lb, 
											problem.ub
										);

		for (const auto & idx : ikid_data.joint_range)
    {
      BOOST_TEST(check_eigen_dense_abs_or_rel_equal(
        ikid_data.vis[idx].toVector(), problem.vis[idx].toVector(), 
				1e-4));
    }

		BOOST_TEST(check_eigen_dense_abs_or_rel_equal(ikid_data.z, problem.z, 1e-4));


	}

} // test_romeo

BOOST_AUTO_TEST_SUITE_END()