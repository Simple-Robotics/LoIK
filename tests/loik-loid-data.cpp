//
// Copyright (c) 2024 INRIA
//

#include "loik/loik-loid-data.hpp"

#include <pinocchio/algorithm/check.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/sample-models.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

#include <iostream>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

using Scalar = double;
using Model = pinocchio::ModelTpl<Scalar>;
using Data = pinocchio::DataTpl<Scalar>;
using IkIdData = loik::IkIdDataTpl<Scalar>;

BOOST_AUTO_TEST_CASE(test_loik_loid_data_creation) {
  // empty model data
  Model empty_model;
  IkIdData empty_data(empty_model);

  // humanoid model data
  Model humanoid_model;
  pinocchio::buildModels::humanoidRandom(humanoid_model);
  Data humanoid_data(humanoid_model);
  IkIdData humanoid_ikid_data(humanoid_model);

  // checkIkIdData(humanoid_model, humanoid_ikid_data, humanoid_data);

  // std::cout << checkIkIdData(humanoid_model, humanoid_ikid_data,
  // humanoid_data) << std::endl;

  // test against pinocchio data
  BOOST_CHECK(checkIkIdData(humanoid_model, humanoid_ikid_data));
  BOOST_CHECK(checkIkIdData(humanoid_model, humanoid_ikid_data, humanoid_data));
}

BOOST_AUTO_TEST_CASE(test_loik_loid_data_copy_and_equal_op) {
  Model humanoid_model;
  pinocchio::buildModels::humanoidRandom(humanoid_model);

  Data humanoid_data(humanoid_model);
  IkIdData humanoid_ikid_data(humanoid_model);
  IkIdData humanoid_ikid_data_copy = humanoid_ikid_data;

  // check ik_id_data against itself and its copy
  BOOST_CHECK(humanoid_ikid_data == humanoid_ikid_data);
  BOOST_CHECK(humanoid_ikid_data == humanoid_ikid_data_copy);

  // check ik_id_data_copy against data
  BOOST_CHECK(humanoid_ikid_data_copy == humanoid_data);
  BOOST_CHECK(humanoid_data == humanoid_ikid_data_copy);

  //
  humanoid_ikid_data_copy.oMi[0].setRandom();
  BOOST_CHECK(humanoid_ikid_data != humanoid_ikid_data_copy);
  BOOST_CHECK(humanoid_ikid_data_copy != humanoid_ikid_data);
  BOOST_CHECK(humanoid_ikid_data_copy != humanoid_data);
  BOOST_CHECK(humanoid_data != humanoid_ikid_data_copy);
}

BOOST_AUTO_TEST_CASE(test_container_aligned_vector) {
  Model model;
  pinocchio::buildModels::humanoidRandom(model);

  IkIdData ikid_data(model);

  pinocchio::container::aligned_vector<IkIdData::Motion> &vis = ikid_data.vis;
  PINOCCHIO_ALIGNED_STD_VECTOR(IkIdData::Motion) &vis_using_macro =
      ikid_data.vis;

  ikid_data.vis[0].setRandom();
  BOOST_CHECK(vis[0] == ikid_data.vis[0]);
  BOOST_CHECK(vis_using_macro[0] == ikid_data.vis[0]);
}

BOOST_AUTO_TEST_CASE(test_std_vector_of_Data) {
  Model model;
  pinocchio::buildModels::humanoidRandom(model);

  PINOCCHIO_ALIGNED_STD_VECTOR(IkIdData) ikid_datas;

  for (size_t k = 0; k < 20; ++k)
    ikid_datas.push_back(IkIdData(model));
}

BOOST_AUTO_TEST_SUITE_END()
