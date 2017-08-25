#include "SolvingRequirements.h"

#include <iostream>
// boost
#define BOOST_TEST_MODULE SolvingRequirementsTest
#include <boost/test/unit_test.hpp>

using namespace tvm;
using namespace Eigen;

bool checkRequirements(const SolvingRequirements& sr,
                       bool defaultPriority, int priority,
                       bool defaultWeight, double weight,
                       bool defaultAWeight, const VectorXd& aweight,
                       bool defaultEval, ViolationEvaluationType type)
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

BOOST_AUTO_TEST_CASE(SolvingRequirementsTest)
{
  SolvingRequirements s0;
  BOOST_CHECK(checkRequirements(s0, true, 0, true, 1, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s1(PriorityLevel(2), Weight(3));
  BOOST_CHECK(checkRequirements(s1, false, 2, false, 3, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s2(PriorityLevel(2), Weight(1));
  BOOST_CHECK(checkRequirements(s2, false, 2, true, 1, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s3(Weight(3), PriorityLevel(2));
  BOOST_CHECK(checkRequirements(s3, false, 2, false, 3, true, VectorXd(), true, ViolationEvaluationType::L2));

  SolvingRequirements s4(AnisotropicWeight((VectorXd(3) << 3,4,5).finished()), Weight(3), ViolationEvaluation(ViolationEvaluationType::L1), PriorityLevel(2));
  BOOST_CHECK(checkRequirements(s4, false, 2, false, 3, false, (VectorXd(3) << 3, 4, 5).finished(), false, ViolationEvaluationType::L1));

  BOOST_CHECK_THROW(SolvingRequirements s(PriorityLevel(-1)), std::runtime_error);

  BOOST_CHECK_THROW(SolvingRequirements s(Weight(-1)), std::runtime_error);

  BOOST_CHECK_THROW(SolvingRequirements s(AnisotropicWeight(Vector3d(-1,2,3))), std::runtime_error);
}
