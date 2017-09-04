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
  template<typename T>
  class SingleSolvingRequirement
  {
  public:
    const T& value() const;
    bool isDefault() const;

  protected:
    SingleSolvingRequirement(const T& val, bool isDefault);

    /** Is this requirement at it default value*/
    bool default_;
    
    T value_;
  };

  /** This class represents the priority level of a constraint*/
  class TVM_DLLAPI PriorityLevel : public SingleSolvingRequirement<int>
  {
  public:
    /** Priority level p>=0*/
    PriorityLevel(int p=0);
  };

  /** This class represents the scalar weight alpha of a constraint, within its 
    * priority level. It is meant to ajust the influence of several constraints 
    * at the same level.
    * Given a scalar weight alpha, and a constraint violation measurement f(x), 
    * the product alpha*f(x) will be minimized.
    */
  class TVM_DLLAPI Weight : public SingleSolvingRequirement<double>
  {
  public:
    Weight(double alpha=1);
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
  class TVM_DLLAPI AnisotropicWeight : public SingleSolvingRequirement<Eigen::VectorXd>
  {
  public:
    AnisotropicWeight();
    AnisotropicWeight(const Eigen::VectorXd& w);
  };


  class TVM_DLLAPI ViolationEvaluation : public SingleSolvingRequirement<ViolationEvaluationType>
  {
  public:
    ViolationEvaluation(ViolationEvaluationType t = ViolationEvaluationType::L2);
  };


  /** This macro adds a member of type T named \a member to a class, and a 
    * method \a \name to access this member
    */
  #define ADD_REQUIREMENT(T, name, member) \
  public: \
    const T& name() const { return member; } \
  private: \
    T member; \
    template<typename ... Args> \
    void build(const T& m, const Args& ... args) \
    { \
      static_assert(!check_args<T, Args...>(), \
                    #T" has already been specified"); \
      member = m; \
      build(args...); \
    }


  /** One class to rule them all.*/
  class TVM_DLLAPI SolvingRequirements
  {
  public:
    template<typename ... Args>
    SolvingRequirements(const Args & ... args)
    {
      build(args...);
    }

  private:
    template<typename T, typename Arg0, typename ... Args>
    static constexpr bool check_args()
    {
      return std::is_same<T, Arg0>::value || check_args<T, Args...>();
    }

    template<typename T>
    static constexpr bool check_args()
    {
      return false;
    }

    ADD_REQUIREMENT(PriorityLevel, priorityLevel, priority_)
    ADD_REQUIREMENT(Weight, weight, weight_)
    ADD_REQUIREMENT(AnisotropicWeight, anisotropicWeight, aWeight_)
    ADD_REQUIREMENT(ViolationEvaluation, violationEvaluation, evalType_)

    void build() {}
  };




  template<typename T>
  bool SingleSolvingRequirement<T>::isDefault() const
  {
    return default_;
  }

  template<typename T>
  SingleSolvingRequirement<T>::SingleSolvingRequirement(const T& val, bool isDefault)
    : default_(isDefault), value_(val)
  {
  }

  template<typename T>
  const T& SingleSolvingRequirement<T>::value() const
  {
    return value_;
  }
}
