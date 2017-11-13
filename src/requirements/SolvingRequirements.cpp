#include <tvm/requirements/SolvingRequirements.h>

namespace tvm
{

namespace requirements
{

  PriorityLevel::PriorityLevel(int p)
    : SingleSolvingRequirement<int>(p, p==0)
  {
    if (p < 0)
      throw std::runtime_error("Priority level must be non-negative.");
  }

  Weight::Weight(double alpha)
    : SingleSolvingRequirement(alpha, alpha==1)
  {
    if (alpha < 0)
      throw std::runtime_error("weight must be non negative.");
  }

  AnisotropicWeight::AnisotropicWeight()
    : SingleSolvingRequirement(Eigen::VectorXd(), true)
  {
  }

  AnisotropicWeight::AnisotropicWeight(const Eigen::VectorXd & w)
    : SingleSolvingRequirement(w, false)
  {
    if ((w.array() < 0).any())
      throw std::runtime_error("weights must be non-negative.");
  }

  ViolationEvaluation::ViolationEvaluation(ViolationEvaluationType t)
    : SingleSolvingRequirement(t, t == ViolationEvaluationType::L2)
  {
  }

}  // namespace requirements

}  // namespace tvm
