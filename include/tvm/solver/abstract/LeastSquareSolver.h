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
/** Base class for a (constrained) least-square solver.
 *
 * The problem to be solved has the general form
 * min. ||Ax-b||^2
 * s.t.      C_e x = d
 *      l <= C_i x <= u
 *         xl <= x <= xu
 *
 * where l or u might be set to -inf or +inf, and the explicit bounds are
 * optionnal.
 *
 * When deriving this class, also remember to derive the factory class
 * LSSolverFactory as well.
 */
class TVM_DLLAPI LeastSquareSolver
{
public:
  LeastSquareSolver(bool verbose = false);
  LeastSquareSolver(const LeastSquareSolver &) = delete;
  LeastSquareSolver & operator=(const LeastSquareSolver &) = delete;
  virtual ~LeastSquareSolver() = default;
  /** Open a build sequence for a problem on the current variables (set
   * through the inherited ProblemComputationData::addVariable) with the
   * specified dimensions, allocating the memory needed.
   *
   * \param x The variables of the problem. The object need to be valid until ::finalizeBuild is called.
   * \param nObj Row size of A.
   * \param nEq Row size of C_e.
   * \param nIneq Row size of C_i.
   * \param useBounds Presence of explicit bounds in the problem.
   * \param subs Possible substitutions used for solving.
   *
   * Once a build is started, objective, constraints and bounds can be added
   * through ::addObjective, ::addConstraint and ::addBound, until
   * ::finalizeBuild is called.
   */
  void startBuild(const VariableVector & x,
                  int nObj,
                  int nEq,
                  int nIneq,
                  bool useBounds = true,
                  const hint::internal::Substitutions * const subs = nullptr);
  /** Finalize the build.*/
  void finalizeBuild();

  /** Add a bound constraint to the solver. If multiple bounds appears on the
   * same variable, their intersection is taken.
   */
  void addBound(LinearConstraintPtr bound);

  /** Add a constraint to the solver. */
  void addConstraint(LinearConstraintPtr cstr);

  /** Add an objective to the solver with given requirements
   * \param obj The linear expression added in least-square form
   * \param req The solving requirements. Only the weight-related requirements
   *   will be taken into account
   * \param additionalWeight An additional factor that will multiply the other weights.
   */
  void addObjective(LinearConstraintPtr obj, SolvingRequirementsPtr req, double additionalWeight = 1);

  /** Set ||x||^2 as the least square objective of the problem.
   * \warning this replace previously added objectives.
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
   * \internal Assumes the vector of variables and substitions are the same as
   * when the problem was built.
   */
  void process(const internal::SolverEvents & se);

protected:
  struct ImpactFromChanges
  {
    bool equalityConstraints_ = false;
    bool inequalityConstraints_ = false;
    bool bounds_ = false;
    bool objectives_ = false;

    template<typename Derived>
    static bool willReallocate(const Eigen::DenseBase<Derived> & M, int rows, int cols = 1);

    ImpactFromChanges operator||(bool b);
    ImpactFromChanges operator||(const ImpactFromChanges & other);
    /** this = this || other*/
    ImpactFromChanges & orAssign(const ImpactFromChanges & other);
  };

  virtual void initializeBuild_(int nObj, int nEq, int nIneq, bool useBounds) = 0;
  virtual ImpactFromChanges resize_(int nObj, int nEq, int nIneq, bool useBounds) = 0;
  virtual void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) = 0;
  virtual void addEqualityConstraint_(LinearConstraintPtr cstr) = 0;
  virtual void addIneqalityConstraint_(LinearConstraintPtr cstr) = 0;
  virtual void addObjective_(LinearConstraintPtr obj, SolvingRequirementsPtr req, double additionalWeight = 1) = 0;
  virtual void setMinimumNorm_() = 0;
  virtual void preAssignmentProcess_() {}
  virtual void postAssignmentProcess_() {}
  virtual bool solve_() = 0;
  virtual const Eigen::VectorXd & result_() const = 0;
  virtual bool handleDoubleSidedConstraint_() const = 0;
  virtual Range nextEqualityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const = 0;
  virtual Range nextInequalityConstraintRange_(const constraint::abstract::LinearConstraint & cstr) const = 0;
  virtual Range nextObjectiveRange_(const constraint::abstract::LinearConstraint & cstr) const = 0;
  /** Remove the bounds on variable at given range from the data passed to the
   * solver (e.g. set the bounds to -/+Inf).
   */
  virtual void removeBounds_(const Range & range) = 0;
  virtual void updateEqualityTargetData(scheme::internal::AssignmentTarget & target) = 0;
  virtual void updateInequalityTargetData(scheme::internal::AssignmentTarget & target) = 0;
  virtual void updateBoundTargetData(scheme::internal::AssignmentTarget & target) = 0;
  virtual void updateObjectiveTargetData(scheme::internal::AssignmentTarget & target) = 0;

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
  int nEq_;
  int nIneq_;
  int nObj_;
  int objSize_;
  int eqSize_;
  int ineqSize_;

private:
  bool buildInProgress_;
  bool verbose_;
  VariableVector const * variables_;
  /** Used to track what is the first bound applied to a given variable, if any. */
  map<Variable *, std::vector<constraint::abstract::LinearConstraint *>> boundsOrder_;
  /** List of assignments used for assembling the problem data. */
  AssignmentVector assignments_;
  /** Keeping tracks of which assignments are associated to a constraint.
   * \todo most of the times, there will be a single assignment per constraint.
   * This would be a good place to use small vector-like container.
   */
  MapToAssignment objectiveToAssignments_;
  MapToAssignment equalityConstraintToAssignments_;
  MapToAssignment inequalityConstraintToAssignments_;
  MapToAssignment boundToAssignments_;
  const hint::internal::Substitutions * subs_;
};

/** A base class for LeastSquareSolver factory.
 *
 * The goal of this class is to be passed to a resolution scheme to specify
 * its underlying solver.
 */
class TVM_DLLAPI LSSolverFactory
{
protected:
  LSSolverFactory(const std::string & solverName) : solverName_(solverName) {}

public:
  virtual ~LSSolverFactory() = default;

  virtual std::unique_ptr<LSSolverFactory> clone() const = 0;
  virtual std::unique_ptr<LeastSquareSolver> createSolver() const = 0;

private:
  std::string solverName_;
};

template<typename... Args>
inline void LeastSquareSolver::addAssignement(Args &&... args)
{
  assignments_.emplace_back(new MarkedAssignment(std::forward<Args>(args)...));
}

template<typename Derived>
inline bool LeastSquareSolver::ImpactFromChanges::willReallocate(const Eigen::DenseBase<Derived> & M,
                                                                 int rows,
                                                                 int cols)
{
  return M.rows() * M.cols() != rows * cols;
}

} // namespace abstract

} // namespace solver

} // namespace tvm
