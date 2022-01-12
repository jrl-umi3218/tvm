/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/hint/internal/Substitutions.h>
#include <tvm/scheme/internal/Assignment.h>
#include <tvm/solver/internal/Option.h>
#include <tvm/solver/internal/SolverEvents.h>
#include <tvm/utils/internal/map.h>

namespace tvm
{

namespace solver
{

namespace abstract
{
/** Base class for a hierarchical least-square solver.
 *
 * The problem to be solved has the general form
 * lex.min. (||v_1||^2, ||w_1||^2, ...., ||v_p||^2, ||w_p||^2)
 * s.t.        A_i x + v_i = b_i   i= 1..p
 *      l_i <= C_i x + w_i <= u_i   i= 1..p
 *
 * where l_i or u_i might be set to -inf or +inf.
 *
 * When deriving this class, also remember to derive the factory class
 * HLSSolverFactory as well.
 */
class TVM_DLLAPI HierarchicalLeastSquareSolver
{
public:
  HierarchicalLeastSquareSolver(bool verbose = false);
  HierarchicalLeastSquareSolver(const HierarchicalLeastSquareSolver &) = delete;
  HierarchicalLeastSquareSolver & operator=(const HierarchicalLeastSquareSolver &) = delete;
  virtual ~HierarchicalLeastSquareSolver() = default;
  /** Open a build sequence for a problem on the current variables (set
   * through the inherited ProblemComputationData::addVariable) with the
   * specified dimensions, allocating the memory needed.
   *
   * \param x The variables of the problem. The object need to be valid until ::finalizeBuild is called.
   * \param nEq For each priority level, the row size of A_i.
   * \param nIneq For each priority level the row size of C_i.
   * \param useBounds Presence of explicit bounds as the first priority level in the problem.
   * \param subs Possible substitutions used for solving.
   *
   * Once a build is started, constraints and bounds can be added
   * through ::addConstraint and ::addBound, until ::finalizeBuild is called.
   */
  void startBuild(const VariableVector & x,
                  const std::vector<int> & nEq,
                  const std::vector<int> & nIneq,
                  bool useBounds,
                  const hint::internal::Substitutions * const subs = nullptr);
  /** Finalize the build.*/
  void finalizeBuild();

  /** Add a bound constraint to the solver at the top priority level. If multiple bounds appears on the
   * same variable, their intersection is taken.
   */
  void addBound(LinearConstraintPtr bound);

  /** Add a constraint to the solver. */
  void addConstraint(int lvl, LinearConstraintPtr cstr);

  /** Set x = 0 as the last priority level of the problem.
   *
   * \warning This will overwrite any other constraint at this level.
   */
  void setMinimumNorm();

  /** Solve the problem
   * \return true upon success of the resolution.
   */
  bool solve();

  /** Get the result of the previous call to solve()*/
  const Eigen::VectorXd & result() const;

  /** Return the constraint size for the solver. This can be different from
   * the actual constraint size if the constraint is a double-sided inequality
   * but the solver only accept simple sided constraints
   */
  int constraintSize(const constraint::abstract::LinearConstraint & c) const;

  /** Update the data according to the events
   *
   * \internal Assumes the vector of variables and substitutions are the same as
   * when the problem was built.
   */
  void process(const internal::SolverEvents & se);

  /** Number of priority levels*/
  int numberOfLevels() const { return useBounds_ ? static_cast<int>(nEq_.size()) + 1 : static_cast<int>(nEq_.size()); }

protected:
  struct ImpactFromChanges
  {
    ImpactFromChanges(int nLvl);
    ImpactFromChanges(const std::vector<bool> & eq, std::vector<bool> & ineq, bool bounds);

    std::vector<bool> equalityConstraints_;
    std::vector<bool> inequalityConstraints_;
    bool bounds_ = false;
    int newLevels_ = 0;

    template<typename Derived>
    static bool willReallocate(const Eigen::DenseBase<Derived> & M, int rows, int cols = 1);

    ImpactFromChanges operator||(bool b);
    ImpactFromChanges operator||(const ImpactFromChanges & other);
    /** this = this || other*/
    ImpactFromChanges & orAssign(const ImpactFromChanges & other);

    bool any() const
    {
      return std::any_of(equalityConstraints_.begin(), equalityConstraints_.end(), [](bool b) { return b; })
             || std::any_of(inequalityConstraints_.begin(), inequalityConstraints_.end(), [](bool b) { return b; })
             || bounds_;
    }
  };

  virtual void initializeBuild_(const std::vector<int> & nEq, const std::vector<int> & nIneq, bool useBounds) = 0;
  virtual ImpactFromChanges resize_(const std::vector<int> & nEq, const std::vector<int> & nIneq, bool useBounds) = 0;
  virtual void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) = 0;
  virtual void addEqualityConstraint_(int lvl, LinearConstraintPtr cstr) = 0;
  virtual void addIneqalityConstraint_(int lvl, LinearConstraintPtr cstr) = 0;
  virtual void setMinimumNorm_() = 0;
  virtual void resetBounds_() = 0;
  virtual void preAssignmentProcess_() {}
  virtual void postAssignmentProcess_() {}
  virtual bool solve_() = 0;
  virtual const Eigen::VectorXd & result_() const = 0;
  virtual bool handleDoubleSidedConstraint_() const = 0;
  virtual Range nextEqualityConstraintRange_(int lvl, const constraint::abstract::LinearConstraint & cstr) const = 0;
  virtual Range nextInequalityConstraintRange_(int lvl, const constraint::abstract::LinearConstraint & cstr) const = 0;
  /** Remove the bounds on variable at given range from the data passed to the
   * solver (e.g. set the bounds to -/+Inf).
   */
  virtual void removeBounds_(const Range & range) = 0;
  virtual void updateEqualityTargetData(int lvl, scheme::internal::AssignmentTarget & target) = 0;
  virtual void updateInequalityTargetData(int lvl, scheme::internal::AssignmentTarget & target) = 0;
  virtual void updateBoundTargetData(scheme::internal::AssignmentTarget & target) = 0;

  /** If for a derived class, the change on a category implies the change on
   * others, \p impact is changed accordingly.
   */
  virtual void applyImpactLogic(ImpactFromChanges & impact);

  virtual void printProblemData_() const = 0;
  virtual void printDiagnostic_() const = 0;

  const VariableVector & variables() const { return *variables_; }
  const hint::internal::Substitutions * substitutions() const { return subs_; }

  template<typename... Args>
  void addAssignement(Args &&... args);

private:
  void updateWeights(const internal::SolverEvents & se);
  bool updateVariables(const internal::SolverEvents & se);
  ImpactFromChanges processRemovedConstraints(const internal::SolverEvents & se);
  ImpactFromChanges previewAddedConstraints(const internal::SolverEvents & se);
  void processAddedConstraints(const internal::SolverEvents & se);

public:
  struct MarkedAssignment
  {
    template<typename... Args>
    MarkedAssignment(Args &&... args) : assignment(std::forward<Args>(args)...), markedForRemoval(false)
    {}
    scheme::internal::Assignment assignment;
    bool markedForRemoval;
  };
  template<typename K, typename T>
  using map = utils::internal::map<K, T>;
  using AssignmentVector = std::vector<std::unique_ptr<MarkedAssignment>>;
  using AssignmentPtrVector = std::vector<MarkedAssignment *>;
  using MapToAssignment = map<constraint::abstract::LinearConstraint *, AssignmentPtrVector>;

protected:
  bool useBounds_ = false;
  std::vector<int> nEq_;
  std::vector<int> nIneq_;
  std::vector<int> eqSize_;
  std::vector<int> ineqSize_;

private:
  bool buildInProgress_;
  bool verbose_;
  VariableVector const * variables_;
  /** List of assignments used for assembling the problem data. */
  AssignmentVector assignments_;
  /** Keeping tracks of which assignments are associated to a constraint.
   * \todo most of the times, there will be a single assignment per constraint.
   * This would be a good place to use small vector-like container.
   */
  std::vector<MapToAssignment> equalityConstraintToAssignments_;
  std::vector<MapToAssignment> inequalityConstraintToAssignments_;
  MapToAssignment boundToAssignments_;
  const hint::internal::Substitutions * subs_;
};

/** A base class for HierarchicalLeastSquareSolver factory.
 *
 * The goal of this class is to be passed to a resolution scheme to specify
 * its underlying solver.
 */
class TVM_DLLAPI HLSSolverFactory
{
protected:
  HLSSolverFactory(const std::string & solverName) : solverName_(solverName) {}

public:
  virtual ~HLSSolverFactory() = default;

  virtual std::unique_ptr<HLSSolverFactory> clone() const = 0;
  virtual std::unique_ptr<HierarchicalLeastSquareSolver> createSolver() const = 0;

private:
  std::string solverName_;
};

template<typename... Args>
inline void HierarchicalLeastSquareSolver::addAssignement(Args &&... args)
{
  assignments_.emplace_back(new MarkedAssignment(std::forward<Args>(args)...));
}

template<typename Derived>
inline bool HierarchicalLeastSquareSolver::ImpactFromChanges::willReallocate(const Eigen::DenseBase<Derived> & M,
                                                                             int rows,
                                                                             int cols)
{
  return M.rows() * M.cols() != rows * cols;
}

} // namespace abstract

} // namespace solver

} // namespace tvm
