#include "SolvingRequirements.h"

#include <array>

namespace tvm
{
  bool SingleSolvingRequirement::isDefault() const
  {
    return default_;
  }

  RequirementType SingleSolvingRequirement::type() const
  {
    return type_;
  }

  SingleSolvingRequirement::SingleSolvingRequirement(RequirementType type, bool isDefault)
    : type_(type), default_(isDefault)
  {
  }

  PriorityLevel::PriorityLevel(int p)
    : SingleSolvingRequirement(RequirementType::PriorityLevel, p==0)
  {
    if (p < 0)
      throw std::runtime_error("Priority level must be non-negative.");
    level_ = p;
  }

  int PriorityLevel::value() const
  {
    return level_;
  }

  Weight::Weight(double alpha)
    : SingleSolvingRequirement(RequirementType::Weight, alpha==1)
  {
    if (alpha < 0)
      throw std::runtime_error("weight must be non negative.");
    alpha_ = alpha;
  }

  double Weight::value() const
  {
    return alpha_;
  }

  AnisotropicWeight::AnisotropicWeight()
    : SingleSolvingRequirement(RequirementType::AnisotropicWeight, true)
  {
  }

  AnisotropicWeight::AnisotropicWeight(const Eigen::VectorXd & w)
    : SingleSolvingRequirement(RequirementType::AnisotropicWeight, false)
  {
    if ((w.array() < 0).any())
      throw std::runtime_error("weights must be non-negative.");
    w_ = w;
  }

  const Eigen::VectorXd & AnisotropicWeight::value() const
  {
    return w_;
  }

  ViolationEvaluation::ViolationEvaluation(ViolationEvaluationType t)
    : SingleSolvingRequirement(RequirementType::ViolationEvaluation, t == ViolationEvaluationType::L2)
  {
    evalType_ = t;
  }

  ViolationEvaluationType ViolationEvaluation::value() const
  {
    return evalType_;
  }

  SolvingRequirements::SolvingRequirements(std::initializer_list<SingleSolvingRequirement> requirements)
  {
    std::array<int, 4> indexes = { -1,-1,-1,-1 };
    auto it = requirements.begin();
    
    // First, we discover the type of requirements present in the initializer list
    // At the end of the loop, indexes[i] contains -1 if requirement RequirementType(i) is not
    // present, and its index in the list otherwise.
    // We throw an error if the same requirement type appears more than once.
    for (int i = 0; i < static_cast<int>(requirements.size()); ++i)
    {
      int id = static_cast<int>((it+i)->type());
      if (indexes[id] < 0)
        indexes[id] = i;
      else
        throw std::runtime_error("You can only give one requirement of each type.");
    }

    // Second, we copy the requirements present. The other ones stay to default
    if (indexes[0] >= 0) priority_ = *static_cast<const PriorityLevel*>      (it + indexes[0]);
    if (indexes[1] >= 0) weight_   = *static_cast<const Weight*>             (it + indexes[1]);
    if (indexes[2] >= 0) aWeight_  = *static_cast<const AnisotropicWeight*>  (it + indexes[2]);
    if (indexes[3] >= 0) evalType_ = *static_cast<const ViolationEvaluation*>(it + indexes[3]);
  }

  const PriorityLevel& SolvingRequirements::priorityLevel() const
  {
    return priority_;
  }

  const Weight& SolvingRequirements::weight() const
  {
    return weight_;
  }

  const AnisotropicWeight& SolvingRequirements::anisotropicWeight() const
  {
    return aWeight_;
  }

  const ViolationEvaluation& SolvingRequirements::violationEvaluation() const
  {
    return evalType_;
  }

}
