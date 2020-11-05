/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/VariableVector.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/scheme/internal/ResolutionSchemeBase.h>
#include <tvm/scheme/internal/SchemeAbilities.h>

namespace tvm
{

class LinearizedControlProblem;

namespace scheme
{

namespace abstract
{

/** Resolution schemes may be defined only for a particular type of problems.
 * We use CRTP for providing a common interface despite this, and
 * performing some basic common operations.
 *
 * The Derived class must provide:
 * - one or several void solve_(ProblemType&, ComputationData&)
 *   methods (several if it handle differently several types of problems).
 * - likewise, one or several createComputationData_(const ProblemType&)
 *   methods returning a std::unique_ptr<ComputationDataType> where
 *   ComputationDataType derives from ProblemComputationData.
 *
 * For a given problem, the solve_ methods is guaranteed to receive the
 * ComputationData instance created by createComputationData_ for the same
 * problem.
 */
template<typename Derived>
class ResolutionScheme : public internal::ResolutionSchemeBase
{
public:
  template<typename Problem>
  bool solve(Problem & problem) const;

  template<typename Problem>
  std::unique_ptr<internal::ProblemComputationData> createComputationData(const Problem & problem) const;

  template<typename Problem>
  void updateComputationData(Problem & problem, internal::ProblemComputationData * data) const;

  /** Returns a reference to the derived object */
  Derived & derived() { return *static_cast<Derived *>(this); }
  /** Returns a const reference to the derived object */
  const Derived & derived() const { return *static_cast<const Derived *>(this); }

protected:
  ResolutionScheme(internal::SchemeAbilities abilities, double big = constant::big_number);
};

/** Base class for scheme solving linear problems
 * For now, it is there only for allowing to differentiate with future
 * non-linear schemes.
 */
template<typename Derived>
class LinearResolutionScheme : public ResolutionScheme<Derived>
{
protected:
  LinearResolutionScheme(internal::SchemeAbilities abilities, double big = constant::big_number);
};

template<typename Derived>
template<typename Problem>
inline bool ResolutionScheme<Derived>::solve(Problem & problem) const
{
  auto data = getComputationData(problem, *this);
  problem.update();
  updateComputationData(problem, data);
  bool b = derived().solve_(problem, data);
  data->setVariablesToSolution();
  problem.substitutions().updateVariableValues();
  return b;
}

template<typename Derived>
template<typename Problem>
inline std::unique_ptr<internal::ProblemComputationData> ResolutionScheme<Derived>::createComputationData(
    const Problem & problem) const
{
  tvm::utils::override_is_malloc_allowed(true);
  return derived().createComputationData_(problem);
  tvm::utils::restore_is_malloc_allowed();
}

template<typename Derived>
template<typename Problem>
void ResolutionScheme<Derived>::updateComputationData(Problem & problem,
                                                      internal::ProblemComputationData * data) const
{
  tvm::utils::override_is_malloc_allowed(true);
  derived().updateComputationData_(problem, data);
  tvm::utils::restore_is_malloc_allowed();
}

template<typename Derived>
inline ResolutionScheme<Derived>::ResolutionScheme(internal::SchemeAbilities abilities, double big)
: ResolutionSchemeBase(abilities, big)
{}

template<typename Derived>
inline LinearResolutionScheme<Derived>::LinearResolutionScheme(internal::SchemeAbilities abilities, double big)
: ResolutionScheme<Derived>(abilities, big)
{}

} // namespace abstract

} // namespace scheme

} // namespace tvm

#include <tvm/scheme/internal/helpers.hpp>
