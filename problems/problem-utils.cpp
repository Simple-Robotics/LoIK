#include "problem-utils.hpp"

namespace loik {

	namespace utils {
		/// General implementation of data loader utils template functions
	
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
		template <typename T>
		void PtreeToDataObj(const ptree& pt, const T& data_obj)
		{
			T& data_obj_out = const_cast<T&>(data_obj);
			data_obj_out = pt.get_value<T>();
		} //PtreeToDataObj


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename T, template<typename...> class Container>
		void PtreeToContainerOfObj(const ptree& pt, const Container<T>& container)
		{
			Container<T>& container_out = const_cast<Container<T>&>(container);
			for (const auto& item : pt) {
				T data_obj = T{};
				PtreeToDataObj(item.second, data_obj);
				container_out.push_back(data_obj);
			}
		} //PtreeToContainerOfObj
		

	} // namespace utils


	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename _Scalar>
	void SequenceDiffIKProblems<_Scalar>::LoadProblemsFromJson(const std::string& file_name)
	{
		Reset();
		std::string problem_data_file_path = file_name;

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

			// name 
			problem_struct.name = time_key;

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
			utils::PtreeToDataObj<DVec>(q_pt, problem_struct.q);

			// H_refs
			const ptree& H_refs_pt = problem_def.get_child("H_refs");
			utils::PtreeToContainerOfObj<Mat6x6, pin_aligned_vec>(H_refs_pt, problem_struct.H_refs);

			// v_refs
			const ptree& v_refs_pt = problem_def.get_child("v_refs");
			utils::PtreeToContainerOfObj<Motion, pin_aligned_vec>(v_refs_pt, problem_struct.v_refs);

			// active_task_constraint_ids
			const ptree& active_task_constraint_ids_pt = problem_def.get_child("active_task_constraint_ids");
			utils::PtreeToContainerOfObj<Index, std::vector>(active_task_constraint_ids_pt, problem_struct.active_task_constraint_ids);

			// Ais
			const ptree& Ais_pt = problem_def.get_child("Ais");
			utils::PtreeToContainerOfObj<Mat6x6, pin_aligned_vec>(Ais_pt, problem_struct.Ais);

			// bis
			const ptree& bis_pt = problem_def.get_child("bis");
			utils::PtreeToContainerOfObj<Vec6, pin_aligned_vec>(bis_pt, problem_struct.bis);

			// lb 
			const ptree& lb_pt = problem_def.get_child("lb");
			utils::PtreeToDataObj<DVec>(lb_pt, problem_struct.lb);


			// ub
			const ptree& ub_pt = problem_def.get_child("ub");
			utils::PtreeToDataObj<DVec>(ub_pt, problem_struct.ub);

			// num_eq_c
			problem_struct.num_eq_c = static_cast<int>(problem_struct.bis.size());


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
			const ptree& vis_pt = problem_sol.get_child("vis");
			utils::PtreeToContainerOfObj<Motion, pin_aligned_vec>(vis_pt, problem_struct.vis);

			// yis
			const ptree& yis_pt = problem_sol.get_child("yis");
			utils::PtreeToContainerOfObj<Vec6, pin_aligned_vec>(yis_pt, problem_struct.yis);

			// w
			const ptree& w_pt = problem_sol.get_child("w");
			utils::PtreeToDataObj<DVec>(w_pt, problem_struct.w);


			// z
			const ptree& z_pt = problem_sol.get_child("z");
			utils::PtreeToDataObj<DVec>(z_pt, problem_struct.z);


			/// push problem to problem_sequence
			problem_sequence.push_back(problem_struct);

		} //endfor

	} // SequenceDiffIKProblems::LoadProblemsFromJson


	/// explicit instantiation and specializations
	namespace utils {
		using Index = typename loik::DiffIKProblem<double>::Index;
		using DVec = typename loik::DiffIKProblem<double>::DVec;
		using Vec6 = typename loik::DiffIKProblem<double>::Vec6; 
		using Motion = typename loik::DiffIKProblem<double>::Motion;
		using Mat6x6 = typename loik::DiffIKProblem<double>::Mat6x6;

		template<typename T>
		using pin_aligned_vec = typename pinocchio::container::aligned_vector<T>;

		// explicit instantiation
		template void PtreeToVec<DVec>(const ptree&, const DVec&);
		template void PtreeToVec<Vec6>(const ptree&, const Vec6&);
		template void PtreeToMat<Mat6x6>(const ptree&, const Mat6x6&);

		
		// `PtreeToDataObj` specializations 
		template <>
		void PtreeToDataObj<DVec>(const ptree& pt, const DVec& vec) { PtreeToVec<DVec>(pt, vec); } // PtreeToDataObj<DVec>
		template <>
		void PtreeToDataObj<Vec6>(const ptree& pt, const Vec6& vec) { PtreeToVec<Vec6>(pt, vec); } // PtreeToDataObj<Vec6>
		template <>
		void PtreeToDataObj<Mat6x6>(const ptree& pt, const Mat6x6& mat) { PtreeToMat<Mat6x6>(pt, mat); } // PtreeToDataObj<Mat6x6>
		template <>
		void PtreeToDataObj<Motion>(const ptree& pt, const Motion& motion)
		{
			Motion& motion_out = const_cast<Motion&>(motion);
			Motion::Vector6 motion_v6 = Motion::Vector6::Zero();
			PtreeToVec(pt, motion_v6);
			motion_out = motion_v6;
		} // PtreeToDataObj<Motion>

		// explicit instantiation
		template void PtreeToDataObj<Index>(const ptree&, const Index&);
		template void PtreeToContainerOfObj<Index, std::vector>(const ptree&, const std::vector<Index>&);
		template void PtreeToContainerOfObj<DVec, pin_aligned_vec>(const ptree&, const pin_aligned_vec<DVec>&);
		template void PtreeToContainerOfObj<Vec6, pin_aligned_vec>(const ptree&, const pin_aligned_vec<Vec6>&);
		template void PtreeToContainerOfObj<Mat6x6, pin_aligned_vec>(const ptree&, const pin_aligned_vec<Mat6x6>&);
		template void PtreeToContainerOfObj<Motion, pin_aligned_vec>(const ptree&, const pin_aligned_vec<Motion>&);

	} // namespace utils

	template struct DiffIKProblem<double>;
	template struct SequenceDiffIKProblems<double>;

} // namespace loik