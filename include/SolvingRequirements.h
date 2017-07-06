#pragma once

#include <Eigen/Core>

#include <tvm/api.h>

namespace tvm
{
  /** Note: when adding a task to a problem, the user will want to specify part
    * or all of the following: priority level, (global) weight, different 
    * weights for each dimension, type of norm to consider (and maybe more in 
    * the future). The method through which tasks are added need then to be 
    * able to accomodate these specifications.
    * The classes in this file are meant to fulfill the following points:
    *  - having strongly typed notions of weight, priority, ...
    *  - leting the user specify only part of those notions and rely on default
    *    value for the others.
    *  - allowing to use any order to give the arguments
    *  - while not having to multiply the overload for the method adding the 
    *    task (there are 65 combinations!) and avoiding heavy variadic template 
    *    shenanigans.
    *
    * FIXME: should we add the notion of row selection here as well ?
    */

  /** The different type of requirements on how to solve a constraint. See the
    * eponym class for more details.
    */
  enum class RequirementType
  {
    PriorityLevel = 0,          //We do rely on the fact that this enumeration starts at zero and is continuous
    Weight,
    AnisotropicWeight,
    ViolationEvaluation
  };

  /** Given a constraint, let the vector v(x) its componentwise violation. For
    * example, for the constraint c(x) = 0, we simply have v(x) = c(x), for
    * c(x) >= b, we have v(x) = max(b-c(x),0). 
    * This enumeration specify how v(x) is made into a scalar measure f(x) of
    * this violation.
    */
  enum class ViolationEvaluationType
  {
    L1,     // f(x) = sum(v(x))
    L2,     // f(x) = v(x)'*v(x)
    LINF    // f(x) = max(v(x))
  };

  /** Class representing the way a constraint has to be solved and how it 
    * interacts with other constraints in term of hierarchical and weighted
    * priorities.
    *
    * This is a base class for the sole purpose of conveniency.
    */
  class TVM_DLLAPI SingleSolvingRequirement
  {
  public:
    bool isDefault() const;
    RequirementType type() const;

  protected:
    SingleSolvingRequirement(RequirementType type, bool isDefault);

    /**Type of requirement*/
    RequirementType type_;
    /** Is this requirement at it default value*/
    bool default_;
    
    int level_;
    double alpha_;
    Eigen::VectorXd w_;
    ViolationEvaluationType evalType_;
  };

  /** This class represents the priority level of a constraint*/
  class TVM_DLLAPI PriorityLevel : public SingleSolvingRequirement
  {
  public:
    /** Priority level p>=0*/
    PriorityLevel(int p=0);

    int value() const;
  };

  /** This class represents the scalar weight alpha of a constraint, within its 
    * priority level. It is meant to ajust the influence of several constraints 
    * at the same level.
    * Given a scalar weight alpha, and a constraint violation measurement f(x), 
    * the product alpha*f(x) will be minimized.
    */
  class TVM_DLLAPI Weight : public SingleSolvingRequirement
  {
  public:
    Weight(double alpha=1);

    double value() const;
  };

  /** This class represents an anisotropic weight to give more or less 
    * importance to the different rows of a constraint. It is given as a vector
    * w. It results in the violation v(x) of the constraint being multiplied by
    * diag(w'), where w' depends on the constraint violation evaluation chosen.
    * w' is such that if w was a uniform vector with all components equal to 
    * alpha, the result would be coherent with using a Weight with value alpha.
    *
    * This class can be redundant with Weight, as having a Weight alpha and an
    * AnisotropicWeight w is the same as having a just an AnisotropicWeight w.
    * As a guideline, it should be used only to discriminate between the rows 
    * of a constraint, while Weight would be used to discriminate between
    * different constraints. As such the "mean" value of w should be 1.
    *
    * This class replaces the notion of dimWeight in Tasks.
    *
    * FIXME Do we want to implement some kind of mechanism for constraints 
    * whose size can change? 
    */
  class TVM_DLLAPI AnisotropicWeight : public SingleSolvingRequirement
  {
  public:
    AnisotropicWeight();
    AnisotropicWeight(const Eigen::VectorXd& w);

    const Eigen::VectorXd& value() const;
  };


  class TVM_DLLAPI ViolationEvaluation : public SingleSolvingRequirement
  {
  public:
    ViolationEvaluation(ViolationEvaluationType t = ViolationEvaluationType::L2);

    ViolationEvaluationType value() const;
  };


  /** One class to rule them all. This*/
  class TVM_DLLAPI SolvingRequirements
  {
  public:
    SolvingRequirements(std::initializer_list<SingleSolvingRequirement> requirements = {});

    const PriorityLevel& priorityLevel() const;
    const Weight& weight() const;
    const AnisotropicWeight& anisotropicWeight() const;
    const ViolationEvaluation& violationEvaluation() const;

  private:
    PriorityLevel       priority_;
    Weight              weight_;
    AnisotropicWeight   aWeight_;
    ViolationEvaluation evalType_;
  };
}
