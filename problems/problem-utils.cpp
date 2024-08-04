#include "problem-utils.hpp"

namespace loik {

	namespace utils {
	
	
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename VecLike>
		void 
		PtreeToVec(const ptree& pt, const VecLike& vec)
		{
			// const cast for output
			VecLike& vec_out = const_cast<VecLike&>(vec);

			std::vector<typename VecLike::Scalar> std_vec;
			for (const auto& item : pt) {
					std_vec.push_back(item.second.get_value<typename VecLike::Scalar>());
			}
			vec_out = Eigen::Map<VecLike>(std_vec.data(), std_vec.size());
		} // PtreeToVec


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename VecLike, template<typename> class VecOfVecLike>
		void 
		PtreeToVecOfVec(const ptree& pt, const VecOfVecLike<VecLike>& vec_of_vec)
		{
			VecOfVecLike<VecLike>& vec_of_vec_out = const_cast<VecOfVecLike<VecLike>&>(vec_of_vec);
			for (const auto& item : pt) {
				VecLike vec;
				PtreeToVec(item.second, vec);
        vec_of_vec_out.push_back(vec);
    	}

		} // PtreeToVecOfVec


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename MatLike>
		void 
		PtreeToMat(const ptree& pt, const MatLike& mat)
		{
			// const cast for output
			MatLike& mat_out = const_cast<MatLike&>(mat);

			std::vector<std::vector<typename MatLike::Scalar>> std_mat;
			for (const auto& row : pt) {
				std::vector<typename MatLike::Scalar> std_vec;
				for (const auto& element : row.second){
					std_vec.push_back(element.second.get_value<typename MatLike::Scalar>());
				}
				std_mat.push_back(std_vec);
			}

			if (std_mat.empty()) {
				std::cerr << "WARNING [utils::PtreeToMat]: `std_mat` is empty !!! " << std::endl;
			}

			for (size_t i = 0; i < std_mat.size(); i++) {
				for (size_t j = 0; j < std_mat[i].size(); j++) {
					mat_out(i, j) = std_mat[i][j];
				}
			}

		} // PtreeToMat
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename MatLike, template<typename> class VecOfMatLike>
		void 
		PtreeToVecOfMat(const ptree& pt, const VecOfMatLike<MatLike>& vec_of_mat)
		{
			VecOfMatLike<MatLike>& vec_of_mat_out = const_cast<VecOfMatLike<MatLike>&>(vec_of_mat);
			for (const auto& item : pt) {
				MatLike mat;
				PtreeToMat(item.second, mat);
        vec_of_mat_out.push_back(mat);
    	}
		} // PtreeToVecOfMat


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, template<typename...> class Container>
		void PtreeToContainerVec(const ptree& pt, const Container<T>& container)
		{
			Container<T>& container_out = const_cast<Container<T>&>(container);
			for (const auto& item : pt) {
				container_out.push_back(item.second.get_value<T>());
			}
		} //PtreeToContainerVec
		

	} // namespace utils


	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename _Scalar>
	void SequenceDiffIKProblems<_Scalar>::LoadProblemsFromJson(const std::string& file_name)
	{
		Reset();
		std::string problem_data_file_path = LOIK_PROBLEM_DATA_DIR + file_name;

		ptree pt;

		try {
			// Read the JSON file into the ptree
			boost::property_tree::read_json(problem_data_file_path, pt);

			// Check if the ptree is empty
			if (pt.empty()) {
					std::cerr << "WARNING [SequenceDiffIKProblems::LoadProblemsFromJson] The ptree is empty!" << std::endl;
					return;
			}

    } catch (const boost::property_tree::json_parser_error& e) {
			std::cerr << "[SequenceDiffIKProblems::LoadProblemsFromJson] Error parsing the JSON file: " << e.what() << std::endl;
    } catch (const std::exception& e) {
			std::cerr << "[SequenceDiffIKProblems::LoadProblemsFromJson] An error occurred: " << e.what() << std::endl;
    }


		for (const auto& item : pt) {
			const std::string& time_key = item.first;
			const ptree& problem = item.second;
			Problem problem_struct;

			const ptree& problem_def = problem.get_child("definition");
			const ptree& problem_sol = problem.get_child("solution");

			/// Parse solver param

			// max_iter
			problem_struct.max_iter = problem_def.get<int>("max_iter");
			
			// rho
			problem_struct.rho = problem_def.get<double>("rho");

			// mu0
			problem_struct.mu0 = problem_def.get<double>("mu0");

			// mu_equality_scale_factor
			problem_struct.mu_equality_scale_factor = problem_def.get<double>("mu_equality_scale_factor");

			// tol_abs
			problem_struct.tol_abs = problem_def.get<double>("tol_abs");

			// tol_rel
			problem_struct.tol_rel = problem_def.get<double>("tol_rel");

			// tol_primal_inf
			problem_struct.tol_primal_inf = problem_def.get<double>("tol_primal_inf");

			// tol_dual_inf
			problem_struct.tol_dual_inf = problem_def.get<double>("tol_dual_inf");

			// tol_tail_solve
			problem_struct.tol_tail_solve = problem_def.get<double>("tol_tail_solve");


			/// Parse problem definition

			// q
			const ptree& q_pt = problem_def.get_child("q");
			utils::PtreeToVec<DVec>(q_pt, problem_struct.q);

			// H_refs
			const ptree& H_refs_pt = problem_def.get_child("H_refs");
			utils::PtreeToVecOfMat<Mat6x6, pin_aligned_vec>(H_refs_pt, problem_struct.H_refs);

			// v_refs

			// active_task_constraint_ids
			const ptree& active_task_constraint_ids_pt = problem_def.get_child("active_task_constraint_ids");
			utils::PtreeToContainerVec<Index, std::vector>(active_task_constraint_ids_pt, problem_struct.active_task_constraint_ids);
			std::cout << "active_task_constraint_ids: " << std::endl;
			for (const auto& id : problem_struct.active_task_constraint_ids) {
				std::cout << "id: " << id << std::endl;
			}

			// Ais
			const ptree& Ais_pt = problem_def.get_child("Ais");
			utils::PtreeToVecOfMat<Mat6x6, pin_aligned_vec>(Ais_pt, problem_struct.Ais);
			std::cout << "Ais: " << std::endl;
			for (const auto& Ai : problem_struct.Ais) {
				std::cout << "Ai: " << Ai << std::endl;
			}

			// bis
			const ptree& bis_pt = problem_def.get_child("bis");
			utils::PtreeToVecOfVec<Vec6, pin_aligned_vec>(bis_pt, problem_struct.bis);
			std::cout << "bis: " << std::endl;
			for (const auto& bi : problem_struct.bis) {
				std::cout << "bi: " << bi.transpose() << std::endl;
			}

			// lb 
			const ptree& lb_pt = problem_def.get_child("lb");
			utils::PtreeToVec<DVec>(lb_pt, problem_struct.lb);

			// ub
			const ptree& ub_pt = problem_def.get_child("ub");
			utils::PtreeToVec<DVec>(ub_pt, problem_struct.ub);



			/// Parse solution info

			// converged
			problem_struct.converged = problem_sol.get<bool>("converged");

			// primal_infeasible
			problem_struct.primal_infeasible = problem_sol.get<bool>("primal_infeasible");

			// dual_infeasible
			problem_struct.dual_infeasible = problem_sol.get<bool>("dual_infeasible");

			// primal_residual
			problem_struct.primal_residual = problem_sol.get<double>("primal_residual");

			// dual_residual
			problem_struct.dual_residual = problem_sol.get<double>("dual_residual");

			// mu
			problem_struct.mu = problem_sol.get<double>("mu");

			// n_iter
			problem_struct.n_iter = problem_sol.get<int>("n_iter");

			// n_tail_solve_iter
			problem_struct.n_tail_solve_iter = problem_sol.get<int>("n_tail_solve_iter");

			// n_active_ineq_constraint

			// vis

			// yis
			const ptree& yis_pt = problem_sol.get_child("yis");
			utils::PtreeToVecOfVec<Vec6, pin_aligned_vec>(yis_pt, problem_struct.yis);
			std::cout << "yis: " << std::endl;
			for (const auto& yi : problem_struct.yis) {
				std::cout << "yi: " << yi.transpose() << std::endl;
			}

			// w
			const ptree& w_pt = problem_sol.get_child("w");
			utils::PtreeToVec<DVec>(w_pt, problem_struct.w);

			// z
			const ptree& z_pt = problem_sol.get_child("z");
			utils::PtreeToVec<DVec>(z_pt, problem_struct.z);


			/// push problem to problem_sequence
			problem_sequence.push_back(problem_struct);

		} //endfor

	} // SequenceDiffIKProblems::LoadProblemsFromJson

	/// explicit instantiation 
	namespace utils {
		using Index = typename loik::DiffIKProblem<double>::Index;
		using DVec = typename loik::DiffIKProblem<double>::DVec;
		using Vec6 = typename loik::DiffIKProblem<double>::Vec6; 
		using Motion = typename loik::DiffIKProblem<double>::Motion;
		using Mat6x6 = typename loik::DiffIKProblem<double>::Mat6x6;

		template<typename T>
		using pin_aligned_vec = typename pinocchio::container::aligned_vector<T>;
		template void PtreeToVec<DVec>(const ptree&, const DVec&);
		template void PtreeToVec<Vec6>(const ptree&, const Vec6&);
		template void PtreeToVecOfVec<Vec6, pin_aligned_vec>(const ptree&, const pin_aligned_vec<Vec6>&);
		template void PtreeToMat<Mat6x6>(const ptree&, const Mat6x6&);
		template void PtreeToVecOfMat<Mat6x6, pin_aligned_vec>(const ptree&, const pin_aligned_vec<Mat6x6>&);
		template void PtreeToContainerVec<Index, std::vector>(const ptree&, const std::vector<Index>&);

	} // namespace utils

	template struct DiffIKProblem<double>;
	template struct SequenceDiffIKProblems<double>;

} // namespace loik