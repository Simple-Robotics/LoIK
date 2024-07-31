#include "problem-utils.hpp"

namespace loik {


	template <typename _Scalar>
	void SequenceDiffIKProblems<_Scalar>::LoadProblemsFromJson(const std::string& file_name)
	{
		std::string problem_data_file_path = LOIK_PROBLEM_DATA_DIR + file_name;

		boost::property_tree::ptree pt;
		boost::property_tree::read_json(problem_data_file_path, pt);


		

	}

	/// explicit instantiation 
	template struct DiffIKProblem<double>;
	template struct SequenceDiffIKProblems<double>;

}