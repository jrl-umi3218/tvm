/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/requirements/AnisotropicWeight.h>
#include <tvm/requirements/PriorityLevel.h>
#include <tvm/requirements/ViolationEvaluation.h>
#include <tvm/requirements/Weight.h>

#include <Eigen/Core>

namespace tvm
{

namespace requirements
{

/** \internal when adding a task to a problem, the user will want to specify
 * part or all of the following: priority level, (global) weight,
 * different weights for each dimension, type of norm to consider (and
 * maybe more in the future). The method through which tasks are added
 * need then to be able to accomodate these specifications.
 *
 * The classes in this file are meant to fulfill the following points:
 *
 *  - having strongly typed notions of weight, priority, ...
 *
 *  - leting the user specify only part of those notions and rely on
 *  default value for the others.
 *
 *  - allowing to use any order to give the arguments
 *
 *  - while not having to multiply the overload for the method adding
 *  the task (there are 65 combinations!).
 *
 * FIXME: should we add the notion of row selection here as well ?
 */

/** This macro adds a member of type \p T named \p member to a class, and a
 * method \p name to access this member.
 */
#define ADD_REQUIREMENT(T, name, member)                                                             \
public:                                                                                              \
  const T<Lightweight> & name() const { return member; }                                             \
  T<Lightweight> & name() { return member; }                                                         \
                                                                                                     \
private:                                                                                             \
  T<Lightweight> member;                                                                             \
  template<typename... Args>                                                                         \
  void build(const T<Lightweight> & m, const Args &... args)                                         \
  {                                                                                                  \
    static_assert(!check_args<T<Lightweight>, Args...>() && !check_args<T<!Lightweight>, Args...>(), \
                  #T " has already been specified");                                                 \
    member = m;                                                                                      \
    build(args...);                                                                                  \
  }                                                                                                  \
  template<typename... Args>                                                                         \
  void build(const T<!Lightweight> & m, const Args &... args)                                        \
  {                                                                                                  \
    static_assert(!check_args<T<Lightweight>, Args...>() && !check_args<T<!Lightweight>, Args...>(), \
                  #T " has already been specified");                                                 \
    member = m;                                                                                      \
    build(args...);                                                                                  \
  }

/** This class groups a PriorityLevel, a Weight, an AnisotropicWeight and a
 * ViolationEvaluation, to describe the way a constraint need to be solved
 * and interact with other constraints.
 */
template<bool Lightweight = true>
class SolvingRequirementsBase
{
public:
  /** Constructor. The arguments can be a PriorityLevel, a Weight, an
   * AnisotropicWeight, or a ViolationEvaluation, or any combination of these
   * objects, in any order, provided an instance of each type appears at most
   * once.
   * If an instance of a type is not given, the default value is taken.
   * \sa PriorityLevel, Weight, AnisotropicWeight, ViolationEvaluation
   */
  template<typename... Args>
  SolvingRequirementsBase(const Args &... args)
  {
    build(args...);
  }

private:
  template<typename T, typename Arg0, typename... Args>
  static constexpr bool check_args()
  {
    return std::is_same<T, Arg0>::value || check_args<T, Args...>();
  }

  template<typename T>
  static constexpr bool check_args()
  {
    return false;
  }

  ADD_REQUIREMENT(PriorityLevelBase, priorityLevel, priority_)
  ADD_REQUIREMENT(WeightBase, weight, weight_)
  ADD_REQUIREMENT(AnisotropicWeightBase, anisotropicWeight, aWeight_)
  ADD_REQUIREMENT(ViolationEvaluationBase, violationEvaluation, evalType_)

  void build() {}
};

class SolvingRequirements : public SolvingRequirementsBase<true>
{
  using SolvingRequirementsBase::SolvingRequirementsBase;
};

class SolvingRequirementsWithCallbacks : public SolvingRequirementsBase<false>
{
public:
  using SolvingRequirementsBase::SolvingRequirementsBase;
  SolvingRequirementsWithCallbacks(const SolvingRequirementsWithCallbacks &) = delete;
  SolvingRequirementsWithCallbacks & operator=(const SolvingRequirementsWithCallbacks &) = delete;

  explicit SolvingRequirementsWithCallbacks(const SolvingRequirements & req)
  : SolvingRequirementsBase(req.priorityLevel(), req.weight(), req.anisotropicWeight(), req.violationEvaluation())
  {}
};

#undef ADD_REQUIREMENT

} // namespace requirements

} // namespace tvm
