/** Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/internal/traits.h>
#include <tvm/scheme/abstract/ResolutionScheme.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/scheme/internal/LinearizedProblemComputationData.h>
#include <tvm/solver/abstract/HierarchicalLeastSquareSolver.h>

namespace tvm
{

namespace scheme
{
/** A set of options for HierarchicalLeastSquares. */
class TVM_DLLAPI HierarchicalLeastSquaresOptions{/** If \a true, a damping task is added after the last level
                                                  */
                                                 TVM_ADD_NON_DEFAULT_OPTION(autoDamping, false)};

/** This class implements the hierarchical least square scheme. */
class TVM_DLLAPI HierarchicalLeastSquares : public abstract::LinearResolutionScheme<HierarchicalLeastSquares>
{
private:
  struct Memory : public internal::LinearizedProblemComputationData
  {
    Memory(int solverId, std::unique_ptr<solver::abstract::HierarchicalLeastSquareSolver> solver);

    void reset(std::unique_ptr<solver::abstract::HierarchicalLeastSquareSolver> solver);

    std::unique_ptr<solver::abstract::HierarchicalLeastSquareSolver> solver;

  protected:
    void setVariablesToSolution_(tvm::internal::VariableCountingVector & x) override;
  };

  const static internal::SchemeAbilities abilities_;

  /** Check if T derives from HLSSolverFactory. */
  template<typename T>
  using isFactory = std::is_base_of<solver::abstract::HLSSolverFactory, T>;
  /** Helper structure to delay the evaluation of isFactory<T::Factory>::value*/
  template<typename T>
  struct FactoryMemberIsFactory
  {
    static constexpr bool value = isFactory<typename T::Factory>::value;
  };
  /** Check if T has a member T::Factory and if so if T::Factory derives from HLSSolverFactory.*/
  template<typename T>
  using isHLSSolverFactoryOption =
      std::conjunction<tvm::internal::has_public_member_type_Factory<T>, FactoryMemberIsFactory<T>>;

public:
  using ComputationDataType = Memory;

  /** Constructor from a HLSSolverFactory
   * \tparam SolverFactory Any class deriving from HLSSolverFactory.
   * \param solverFactory A factory for the solver that should be used in the resolution scheme.
   * \param schemeOptions Options for the schemes. See tvm::scheme::HierarchicalLeastSquaresOptions.
   */
  template<class SolverFactory, typename std::enable_if<isFactory<SolverFactory>::value, int>::type = 0>
  HierarchicalLeastSquares(const SolverFactory & solverFactory, HierarchicalLeastSquaresOptions schemeOptions = {})
  : LinearResolutionScheme<HierarchicalLeastSquares>(abilities_), options_(schemeOptions),
    solverFactory_(solverFactory.clone())
  {}

  /** Constructor from a configuration class
   * \tparam SolverOptions Any class representing solver options. The class must have a
   *    member type \a Factory referring to a class C deriving from HLSSolverFactory
   *    and such that C can be constructed from SolverOptions.
   * \param solverOptions A set of options for the solver to be used by the resolution scheme.
   * \param schemeOptions Options for the scheme. See tvm::Scheme::HierarchicalLeastSquaresOptions.
   */
  template<class SolverOptions, typename std::enable_if<isHLSSolverFactoryOption<SolverOptions>::value, int>::type = 0>
  HierarchicalLeastSquares(const SolverOptions & solverOptions, HierarchicalLeastSquaresOptions schemeOptions = {})
  : HierarchicalLeastSquares(typename SolverOptions::Factory(solverOptions), schemeOptions)
  {}

  /** A fallback constructor that is enabled when none of the others are.
   * It always fails at compilation time to provide a nice error message.
   */
  template<typename T,
           typename std::enable_if<!isFactory<T>::value && !isHLSSolverFactoryOption<T>::value, int>::type = 0>
  HierarchicalLeastSquares(const T &, HierarchicalLeastSquaresOptions = {})
  : LinearResolutionScheme<HierarchicalLeastSquares>(abilities_)
  {
    static_assert(tvm::internal::always_false<T>::value,
                  "First argument can only be a HLSSolverFactory or a solver configuration. "
                  "A configuration needs to have a Factory member type that is itself deriving from HLSSolverFactory. "
                  "See LexLSHLSSolverOptions for an example.");
  }

  /** \internal Copy and move are deleted because of the unique_ptr members of
   * the class on polymorphic types. If these semantics are needed, it should
   * be possible to implement them with good care.
   */
  HierarchicalLeastSquares(const HierarchicalLeastSquares &) = delete;
  HierarchicalLeastSquares(HierarchicalLeastSquares &&) = delete;
  HierarchicalLeastSquares & operator=(const HierarchicalLeastSquares &) = delete;
  HierarchicalLeastSquares & operator=(HierarchicalLeastSquares &&) = delete;

  /** Private interface for CRTP*/
  bool solve_(const LinearizedControlProblem & problem, internal::ProblemComputationData * data) const;
  void updateComputationData_(const LinearizedControlProblem & problem, internal::ProblemComputationData * data) const;
  std::unique_ptr<Memory> createComputationData_(const LinearizedControlProblem & problem) const;

protected:
  void resetComputationData(const LinearizedControlProblem & problem, Memory * memory) const;
  void processProblem(const LinearizedControlProblem & problem, Memory * memory) const;

  void addTask(const LinearizedControlProblem & problem,
               Memory * memory,
               const TaskWithRequirements & task,
               solver::internal::SolverEvents & se) const;
  void removeTask(const LinearizedControlProblem & problem,
                  Memory * memory,
                  const TaskWithRequirements & task,
                  solver::internal::SolverEvents & se) const;

  HierarchicalLeastSquaresOptions options_;
  /** The factory to create solvers attached to each problem. */
  std::unique_ptr<solver::abstract::HLSSolverFactory> solverFactory_;
};

} // namespace scheme

} // namespace tvm
