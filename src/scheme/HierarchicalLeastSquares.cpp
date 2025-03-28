/* Copyright 2022 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/scheme/HierarchicalLeastSquares.h>

#include <tvm/LinearizedControlProblem.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/solver/internal/SolverEvents.h>
#include <tvm/utils/internal/map.h>

#include <iostream>

namespace tvm
{

namespace scheme
{
using namespace internal;
using VET = requirements::ViolationEvaluationType;

const internal::SchemeAbilities HierarchicalLeastSquares::abilities_ = {NoLimit,
                                                                        {{GeneralLevel, {true, {VET::L2}}}},
                                                                        false};

bool HierarchicalLeastSquares::solve_(const LinearizedControlProblem & problem,
                                      internal::ProblemComputationData * data) const
{
  if(problem.size() > static_cast<int>(problem.substitutions().substitutions().size()))
  {
    Memory * memory = static_cast<Memory *>(data);
    return memory->solver->solve();
  }
  else
  {
    problem.variables().setZero();
    return true;
  }
}

void HierarchicalLeastSquares::updateComputationData_(const LinearizedControlProblem & problem,
                                                      internal::ProblemComputationData * data) const
{
  using EventType = ProblemDefinitionEvent::Type;
  solver::internal::SolverEvents se;

  if(!data->hasEvents())
  {
    return;
  }

  Memory * memory = static_cast<Memory *>(data);

  // If some events require to rebuild the data, we skip all other events.
  for(const auto & e : memory->events())
  {
    if(e.type() == EventType::SubstitutionAddition || e.type() == EventType::SubstitutionRemoval)
    {
      resetComputationData(problem, memory);
      return;
    }
  }
  while(memory->hasEvents())
  {
    auto e = memory->popEvent();
    switch(e.type())
    {
      case EventType::WeightChange: {
        const auto & c = problem.constraintWithRequirements(e.typedEmitter<EventType::WeightChange>());
        se.addScalarWeightEvent(c.constraint.get(), c.requirements->priorityLevel().value());
      }
      break;
      case EventType::AnisotropicWeightChange: {
        const auto & c = problem.constraintWithRequirements(e.typedEmitter<EventType::AnisotropicWeightChange>());
        se.addVectorWeightEvent(c.constraint.get(), c.requirements->priorityLevel().value());
      }
      break;
      case EventType::TaskAddition:
        addTask(problem, memory, e.typedEmitter<EventType::TaskAddition>(), se);
        break;
      case EventType::TaskRemoval:
        removeTask(problem, memory, e.typedEmitter<EventType::TaskRemoval>(), se);
        break;
      default:
        throw std::runtime_error("[HierarchicalLeastSquares::updateComputationData_] Unimplemented event handling.");
    }
  }

  memory->variables(); // update variable vector if needed
  memory->solver->process(se);
}

std::unique_ptr<HierarchicalLeastSquares::Memory> HierarchicalLeastSquares::createComputationData_(
    const LinearizedControlProblem & problem) const
{
  auto memory = std::unique_ptr<Memory>(new Memory(id(), solverFactory_->createSolver()));
  processProblem(problem, memory.get());

  return memory;
}

void HierarchicalLeastSquares::resetComputationData(const LinearizedControlProblem & problem, Memory * memory) const
{
  memory->reset(solverFactory_->createSolver());
  processProblem(problem, memory);
}

void HierarchicalLeastSquares::processProblem(const LinearizedControlProblem & problem, Memory * memory) const
{
  auto & solver = *memory->solver;

  const auto & constraints = problem.constraints();
  const auto & subs = problem.substitutions();
  memory->addConstraints(problem.constraintMap());

  std::vector<LinearConstraintWithRequirements> constr;
  std::vector<LinearConstraintWithRequirements> bounds;

  // scanning constraints
  std::vector<int> nEq;
  std::vector<int> nIneq;
  for(const auto & c : constraints)
  {
    // If the constraint is used for the substitutions, we skip it.
    if(subs.uses(c.constraint))
      continue;
    abilities_.check(c.constraint, c.requirements); // FIXME: should be done in a parent class
    for(const auto & xi : c.constraint->variables())
      memory->addVariable(subs.substitute(xi));

    int p = c.requirements->priorityLevel().value();
    if(canBeUsedAsBound(c.constraint, subs, constraint::Type::DOUBLE_SIDED) && p == 0)
    {
      bounds.push_back(c);
    }
    else
    {
      if(static_cast<size_t>(p) >= nEq.size())
      {
        nEq.resize(p + 1, 0);
        nIneq.resize(p + 1, 0);
      }
      if(c.constraint->isEquality())
        nEq[p] += solver.constraintSize(*c.constraint);
      else
        nIneq[p] += solver.constraintSize(*c.constraint);
      constr.push_back(c);
    }
  }
  // we need to add the additional constraints due to the substitutions.
  // They are added at level 0.
  if(subs.additionalConstraints().size() > 0 && nEq.size() == 0)
  {
    nEq.push_back(0);
    nIneq.push_back(0);
  }
  for(const auto & c : subs.additionalConstraints())
  {
    nEq[0] += c->size();
    constr.push_back(
        {c, std::make_shared<requirements::SolvingRequirementsWithCallbacks>(requirements::PriorityLevel(0)), false});
  }

  if(options_.autoDamping().value())
  {
    nEq.push_back(memory->variables().totalSize());
    nIneq.push_back(0);
  }

  // configure assignments. FIXME: can we find a better way ?
  Assignment::big_ = big_number_;

  // allocating memory for the solver
  solver.startBuild(memory->variables(), nEq, nIneq, bounds.size() > 0, &subs);

  // assignments for general constraints
  for(const auto & c : constr)
  {
    solver.addConstraint(c.constraint, c.requirements);
  }

  if(options_.autoDamping().value())
    solver.setMinimumNorm();

  // assignments for bounds
  for(const auto & b : bounds)
  {
    solver.addBound(b.constraint);
  }

  solver.finalizeBuild();
}

void HierarchicalLeastSquares::addTask(const LinearizedControlProblem & problem,
                                       Memory * memory,
                                       const TaskWithRequirements & task,
                                       solver::internal::SolverEvents & se) const
{
  // We add a task that is not in the computation data. We get the constraint from problem.
  // However this task might have been removed from problem after being added (but before
  // the computation data have been updated). If this is the case, we skip the addition.
  auto optc = problem.constraintWithRequirementsNoThrow(task);
  if(!optc)
    return;

  // If there is really a task to be added, we need to record the mapping in memory
  auto c = optc->get();
  memory->addConstraint(task, c);
  const auto & subs = problem.substitutions();

  abilities_.check(c.constraint, c.requirements);
  for(const auto & xi : c.constraint->variables())
  {
    auto s = subs.substitute(xi);
    for(const auto & si : s)
    {
      if(memory->addVariable(si))
        se.addVariable(si);
    }
  }

  int p = task.requirements.priorityLevel().value();
  if((p == 0) && canBeUsedAsBound(c.constraint, subs, constraint::Type::DOUBLE_SIDED))
  {
    se.addBound(c.constraint);
  }
  else
  {
    se.addConstraint(c.constraint, c.requirements);
  }
}

void HierarchicalLeastSquares::removeTask(const LinearizedControlProblem & problem,
                                          Memory * memory,
                                          const TaskWithRequirements & task,
                                          solver::internal::SolverEvents & se) const
{
  // We need to remove the constraint that was last added for the task.
  // We get this info from memory. It the task is not present, it's because we
  // skipped its addition in addTask before, so wee skip the removal as well.
  auto optc = memory->constraintNoThrow(task);
  if(!optc)
    return;

  const auto & c = optc->get();
  const auto & subs = problem.substitutions();

  if(subs.uses(c.constraint))
    throw std::runtime_error(
        "[WeightedLeastSquares::removeTask]: You cannot remove a constraint used for a substitution.");

  for(const auto & xi : c.constraint->variables())
  {
    auto s = subs.substitute(xi);
    for(const auto & si : s)
    {
      if(memory->removeVariable(si.get()))
        se.removeVariable(si);
    }
  }

  int p = task.requirements.priorityLevel().value();
  if((p == 0) && canBeUsedAsBound(c.constraint, subs, constraint::Type::DOUBLE_SIDED))
  {
    se.removeBound(c.constraint);
  }
  else
  {
    se.removeConstraint(c.constraint, p);
  }
  memory->removeConstraint(task);
}

HierarchicalLeastSquares::Memory::Memory(int solverId,
                                         std::unique_ptr<solver::abstract::HierarchicalLeastSquareSolver> solver)
: LinearizedProblemComputationData(solverId), solver(std::move(solver))
{}

void HierarchicalLeastSquares::Memory::reset(std::unique_ptr<solver::abstract::HierarchicalLeastSquareSolver> solver)
{
  LinearizedProblemComputationData::reset();
  this->solver = std::move(solver);
}

void HierarchicalLeastSquares::Memory::setVariablesToSolution_(tvm::internal::VariableCountingVector & x)
{
  x.set(solver->result());
}

} // namespace scheme

} // namespace tvm
