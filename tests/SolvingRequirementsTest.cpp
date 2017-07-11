#include "SolvingRequirements.h"

#include <iostream>

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

void testSolvingRequirements()
{
  SolvingRequirements s0;
  std::cout << "valid s0: " <<
    checkRequirements(s0, true, 0, true, 1, true, VectorXd(), true, ViolationEvaluationType::L2)
    <<std::endl;

  SolvingRequirements s1({ PriorityLevel(2), Weight(3) });
  std::cout << "valid s1: " <<
    checkRequirements(s1, false, 2, false, 3, true, VectorXd(), true, ViolationEvaluationType::L2)
    << std::endl;

  SolvingRequirements s2({ PriorityLevel(2), Weight(1) });
  std::cout << "valid s2: " <<
    checkRequirements(s2, false, 2, true, 1, true, VectorXd(), true, ViolationEvaluationType::L2)
    << std::endl;

  SolvingRequirements s3({ Weight(3), PriorityLevel(2) });
  std::cout << "valid s3: " <<
    checkRequirements(s3, false, 2, false, 3, true, VectorXd(), true, ViolationEvaluationType::L2)
    << std::endl;

  SolvingRequirements s4({ AnisotropicWeight((VectorXd(3) << 3,4,5).finished()), Weight(3), ViolationEvaluation(ViolationEvaluationType::L1), PriorityLevel(2) });
  std::cout << "valid s4: " <<
    checkRequirements(s4, false, 2, false, 3, false, (VectorXd(3) << 3, 4, 5).finished(), false, ViolationEvaluationType::L1)
    << std::endl;

  try
  {
    SolvingRequirements s({ PriorityLevel(1), PriorityLevel(2) });
  }
  catch (const std::exception & e)
  {
    std::cout << "catch expected exception: " << e.what() << std::endl;
  }

  try
  {
    SolvingRequirements s({ PriorityLevel(-1) });
  }
  catch (const std::exception & e)
  {
    std::cout << "catch expected exception: " << e.what() << std::endl;
  }

  try
  {
    SolvingRequirements s({ Weight(-1) });
  }
  catch (const std::exception & e)
  {
    std::cout << "catch expected exception: " << e.what() << std::endl;
  }

  try
  {
    SolvingRequirements s({ AnisotropicWeight(Vector3d(-1,2,3)) });
  }
  catch (const std::exception & e)
  {
    std::cout << "catch expected exception: " << e.what() << std::endl;
  }
}
