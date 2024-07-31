#pragma once

#include "loik/fwd.hpp"
#include "loik/loik-loid-data-optimized.hpp"
#include "problems/problem-utils.hpp"


int main(int argc, char** argv)
{
    
    using Scalar = double;
    loik::SequenceDiffIKProblems<Scalar> problems;

    std::string problem_data_file_name = "/panda_problems.json";
    problems.LoadProblemsFromJson(problem_data_file_name);

    // typedef loik::IkIdDataTypeOptimizedTpl<Scalar, 0, pinocchio::JointCollectionDefaultTpl> DummyData;


    // typedef DummyData::Scalar NewScalar;

}

