//
// Copyright (c) 2024 INRIA
//

#include "loik/fwd.hpp"
#include "loik/loik-loid-data-optimized.hpp"
#include "loik/loik-loid.hpp"

#ifdef ENABLE_EXTENDED_TESTING
	#include "problems/problem-utils.hpp"
#endif

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/utils/check.hpp>
#include <pinocchio/utils/timer.hpp>

using Scalar = double;
using Model = typename loik::IkIdDataTypeOptimizedTpl<Scalar>::Model;
using IkIdDataOptimized = loik::IkIdDataTypeOptimizedTpl<Scalar>;
using MuUpdateStrat = loik::ADMMPenaltyUpdateStrat;
using FirstOrderLoikOptimized = loik::FirstOrderLoikOptimizedTpl<Scalar>;

IKID_DATA_TYPEDEF_TEMPLATE(IkIdDataOptimized);

struct ProblemSetup
{

  ProblemSetup()
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

}; // struct ProblemSetup


enum SolveType
{
	FULL = 0,
	REPEAT = 1,
	TAILORED = 3
}; // enum SolveType


void test_1st_order_loik_timing(ProblemSetup& problem, const SolveType& solve_type)
{
	problem.max_iter = 2;
  problem.bound_magnitude = 1.0;
  problem.lb = -problem.bound_magnitude * DVec::Ones(problem.robot_model.nv);
  problem.ub = problem.bound_magnitude * DVec::Ones(problem.robot_model.nv);

  IkIdDataOptimized ikid_data(problem.robot_model, problem.num_eq_c);

  FirstOrderLoikOptimized LoikSolver{
    problem.max_iter,
    problem.tol_abs,
    problem.tol_rel,
    problem.tol_primal_inf,
    problem.tol_dual_inf,
    problem.rho,
    problem.mu,
    problem.mu_equality_scale_factor,
    problem.mu_update_strat,
    problem.num_eq_c,
    problem.eq_c_dim,
    problem.robot_model,
    ikid_data,
    problem.warm_start,
    problem.tol_tail_solve,
    problem.verbose,
    problem.logging};

  PinocchioTicToc timer(PinocchioTicToc::US);

#ifdef NDEBUG
  const int NBT = 100000;
// const int NBT = 1;
#else
  const int NBT = 100000;
  std::cout << "(the time score in debug mode is not relevant) " << std::endl;
#endif

  LoikSolver.SolveInit(problem.q, problem.H_ref, problem.v_ref, problem.active_task_constraint_ids, problem.Ais, problem.bis, problem.lb, problem.ub);
  LoikSolver.Solve();
  int iter_took_to_solver = LoikSolver.get_iter();
  std::cout << "Timing over " << iter_took_to_solver << " iterations for solver to solve"
            << std::endl;

	if (solve_type == SolveType::FULL) {
		timer.tic();
		SMOOTH(NBT)
		{
			LoikSolver.Solve(problem.q, problem.H_ref, problem.v_ref, problem.active_task_constraint_ids, problem.Ais, problem.bis, problem.lb, problem.ub);
		}
		std::cout << "LOIK = \t\t\t\t";
		timer.toc(std::cout, NBT);

	} else if (solve_type == SolveType::REPEAT) {
		timer.tic();
		SMOOTH(NBT)
		{
			LoikSolver.Solve();
		}
		std::cout << "LOIK = \t\t\t\t";
		timer.toc(std::cout, NBT);
	} else if (solve_type == SolveType::TAILORED) {
		timer.tic();
		SMOOTH(NBT)
		{
			LoikSolver.Solve(problem.q, problem.active_task_constraint_ids[0], problem.Ais[0], problem.bis[0]);
		}
		std::cout << "LOIK = \t\t\t\t";
		timer.toc(std::cout, NBT);
	} else {
		throw(std::runtime_error("[test_1st_order_loik_timing]: SolveType not supported."));
	}

	// sanity check, solve iteration should be unchanged
	if (LoikSolver.get_iter() != iter_took_to_solver) {
		throw(std::runtime_error("[test_1st_order_loik_timing]: number of iterations to `Solve` has changed."));
	}

}; // test_1st_order_loik_timing


int main(int argc, char** argv)
{
    
	ProblemSetup problem{};

  /// test_1st_order_loik_timing
	test_1st_order_loik_timing(problem, SolveType::REPEAT);

	/// test_1st_order_loik_tailored_timing
	test_1st_order_loik_timing(problem, SolveType::TAILORED);

}

