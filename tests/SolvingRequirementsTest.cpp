/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/requirements/SolvingRequirements.h>

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm::requirements;
using namespace Eigen;

bool checkRequirements(const SolvingRequirements & sr,
                       bool defaultPriority,
                       int priority,
                       bool defaultWeight,
                       double weight,
                       bool defaultAWeight,
                       const VectorXd & aweight,
                       bool defaultEval,
                       ViolationEvaluationType type)
{
  bool b = (sr.priorityLevel().isDefault() == defaultPriority);
  b = b && (sr.priorityLevel().value() == priority);
  b = b && (sr.weight().isDefault() == defaultWeight);
  b = b && (sr.weight().value() == weight);
  b = b && (sr.anisotropicWeight().isDefault() == defaultAWeight);
  b = b && (sr.anisotropicWeight().value() == aweight);
  b = b && (sr.violationEvaluation().isDefault() == defaultEval);
  b = b && (sr.violationEvaluation().value() == type);
  return b;
}

TEST_CASE("Test SolvingRequirements")
{
  SolvingRequirements s0;
  FAST_CHECK_UNARY(checkRequirements(s0, true, 0, true, 1, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s1(PriorityLevel(2), Weight(3));
  FAST_CHECK_UNARY(checkRequirements(s1, false, 2, false, 3, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s2(PriorityLevel(2), Weight(1));
  FAST_CHECK_UNARY(checkRequirements(s2, false, 2, false, 1, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s3(Weight(3), PriorityLevel(2));
  FAST_CHECK_UNARY(checkRequirements(s3, false, 2, false, 3, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s4(AnisotropicWeight((VectorXd(3) << 3, 4, 5).finished()), Weight(3),
                         ViolationEvaluation(ViolationEvaluationType::L1), PriorityLevel(2));
  FAST_CHECK_UNARY(checkRequirements(s4, false, 2, false, 3, false, (VectorXd(3) << 3, 4, 5).finished(), false,
                                     ViolationEvaluationType::L1));

  CHECK_THROWS_AS(SolvingRequirements s(PriorityLevel(-1)), std::runtime_error);

  CHECK_THROWS_AS(SolvingRequirements s(Weight(-1)), std::runtime_error);

  CHECK_THROWS_AS(SolvingRequirements s(AnisotropicWeight(Vector3d(-1, 2, 3))), std::runtime_error);
}
