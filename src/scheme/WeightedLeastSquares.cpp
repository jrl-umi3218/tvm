/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/scheme/WeightedLeastSquares.h>

#include <tvm/LinearizedControlProblem.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/graph/internal/Logger.h>
#include <tvm/solver/internal/SolverEvents.h>
#include <tvm/utils/internal/map.h>

using Logger = tvm::graph::internal::Logger;

namespace tvm
{

namespace scheme
{
using namespace internal;
using VET = requirements::ViolationEvaluationType;

const internal::SchemeAbilities WeightedLeastSquares::abilities_ = {2,
                                                                    {{0, {true, {VET::L2}}}, {1, {false, {VET::L2}}}},
                                                                    true};

bool WeightedLeastSquares::solve_(const LinearizedControlProblem & problem,
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

void WeightedLeastSquares::updateComputationData_(const LinearizedControlProblem & problem,
                                                  internal::ProblemComputationData * data) const
{
  using EventType = ProblemDefinitionEvent::Type;
  solver::internal::SolverEvents se;

  bool rebuildProblem = false;
  // std::cout << "Running updateComputationData_ !" << std::endl;

  if(data->hasEvents())
  {
    Memory * memory = static_cast<Memory *>(data);
    // std::cout << "There are: " << memory->events().size() << " events" << std::endl;

    // If some events require to rebuild the data, we skip all other events.
    // FIXME this skipping breaks in case of new variables addition or removal (TaskUpdateVariables)
    // because resetComputationData calls canBeUsedAsBound, which calls isBound, calling the variables vector
    // of the constraint, which might not be synced with the function
    for(const auto & e : memory->events())
    {
      if(e.type() == EventType::SubstitutionAddition || e.type() == EventType::SubstitutionRemoval)
      {
        // std::cout << "There is a substitution, skipping all events !" << std::endl;
        resetComputationData(problem, memory);
        return;
      }
    }
    // std::cout << "Update comp data: there are " << memory->events().size() << " events to process" << std::endl;
    while(memory->hasEvents())
    {
      auto e = memory->popEvent();
      // std::cout << "Pop ProblemDefinitionEvent" << std::endl;
      switch(e.type())
      {
        case EventType::WeightChange: {
          // std::cout << "Firing weight change event !" << std::endl;
          const auto & c = problem.constraintWithRequirements(e.typedEmitter<EventType::WeightChange>());
          if(c.requirements->priorityLevel().value() == 0)
            throw std::runtime_error(
                "[WeightedLeastSquares::updateComputationData_] "
                "WeightedLeastSquares does not allow to change the weight of a Task with priority 0.");
          se.addScalarWeightEvent(c.constraint.get());
        }
        break;
        case EventType::AnisotropicWeightChange: {
          // std::cout << "Firing anisotropic weight change event !" << std::endl;
          const auto & c = problem.constraintWithRequirements(e.typedEmitter<EventType::AnisotropicWeightChange>());
          if(c.requirements->priorityLevel().value() == 0)
            throw std::runtime_error(
                "[WeightedLeastSquares::updateComputationData_] "
                "WeightedLeastSquares does not allow to change the weight of a Task with priority 0.");
          se.addVectorWeightEvent(c.constraint.get());
        }
        break;
        case EventType::TaskAddition: {
          // std::cout << "Firing task addition event !" << std::endl;
          addTask(problem, memory, e.typedEmitter<EventType::TaskAddition>(), se);
        }
        break;
        case EventType::TaskRemoval: {
          // std::cout << "Firing task removal event !" << std::endl;
          removeTask(problem, memory, e.typedEmitter<EventType::TaskRemoval>(), se);
        }
        break;
        case EventType::TaskUpdate: {
          // Handles the case where a variable was added or removed to the function after the problem has been created
          // e.g calling addVariable in Function::updateJacobian
          // std::cout << "Firing task update variable event !" << std::endl;
          auto & task = e.typedEmitter<EventType::TaskUpdate>();

          // XXX remove re add is insufficient because it pulls the variables from the constraints in the problem,
          // that were built when the task were added so do not hold any new variables.
          // removeTask(problem, memory, task, se);
          // addTask(problem, memory, task, se);

          // Instead we call updateVariables on the task constraint which updates the variables of the constraint
          // according to those of the function

          // std::cout << "remove task " << Logger::logger().typeName(task.task.function().get()) << std::endl;
          removeTask(problem, memory, task, se);
          // Fully recreate the linearizedtaskconstraint from the task
          {
            // auto cstr = problem.constraintNoThrow(task);
            // std::cout << "checking constraint variables before update: " << std::endl;
            // for(const auto & var : cstr->variables())
            // {
            //   std::cout << "var: " << var->name() << std::endl;
            // }
            const_cast<LinearizedControlProblem &>(problem).updateConstraint(task);
            // cstr = problem.constraintNoThrow(task);
            // std::cout << "checking constraint variables after update" << std::endl;
            // for(const auto & var : cstr->variables())
            // {
            //   std::cout << "var: " << var->name() << std::endl;
            // }
          }
          // // XXX: do we need to then update it in memory?
          //
          // static_cast<tvm::constraint::internal::LinearizedTaskConstraint *>(problem.constraint(task).get())
          //      ->updateVariables();
          addTask(problem, memory, task, se);
          const_cast<LinearizedControlProblem &>(problem).update();
          rebuildProblem = true;

          // FIXME Running this skips other events in the queue as it resets the memory events as well, so some variable
          // updates are missing, but remove and add on the task do not rebuild enough, find the necessary steps in
          // processProblem
          // resetComputationData(problem, memory);

          // FIXME: Ideally we would want to only add the new variable,
          // however the following seems to fail on matrixAssignment when subvariables are
          // involved (even if they are not added directly)
          //
          // std::cout << "WeightedLeastSquares: Handling EventType::TaskAddVariable for function: "
          //           << task.task.function()->UpdateBaseName << std::endl;
          // std::cout << "The function has the following variables:" << std::endl;

          // for(const auto & var : task.task.function()->variables())
          // {
          //   // std::cout << var->name() << std::endl;
          //   // se.addVariable(var);
          // }

          break;
        }
        default:
          throw std::runtime_error("[WeightedLeastSquares::updateComputationData_] Unimplemented event handling.");
      }
      // std::cout << "Finished one event" << std::endl;
    }

    memory->variables(); // update variable vector if needed
    memory->solver->process(se);

    if(rebuildProblem)
    {
      processProblem(problem, memory);
    }
  }
}

std::unique_ptr<WeightedLeastSquares::Memory> WeightedLeastSquares::createComputationData_(
    const LinearizedControlProblem & problem) const
{
  auto memory = std::unique_ptr<Memory>(new Memory(id(), solverFactory_->createSolver()));
  processProblem(problem, memory.get());

  return memory;
}

void WeightedLeastSquares::resetComputationData(const LinearizedControlProblem & problem, Memory * memory) const
{
  memory->reset(solverFactory_->createSolver());
  processProblem(problem, memory);
}

void WeightedLeastSquares::processProblem(const LinearizedControlProblem & problem, Memory * memory) const
{
  // std::cout << "WeightedLeastSquares processing problem" << std::endl;
  auto & solver = *memory->solver;

  const auto & constraints = problem.constraints();
  const auto & subs = problem.substitutions();
  memory->addConstraints(problem.constraintMap());

  std::vector<LinearConstraintWithRequirements> constr;
  std::vector<LinearConstraintWithRequirements> bounds;

  // scanning constraints
  int nEq = 0;
  int nIneq = 0;
  int nObj = 0;
  int maxp = 0;

  auto typelambda = [](auto type) {
    switch(type)
    {
      case tvm::constraint::Type::EQUAL:
        return "EQUAL";
        break;
      case tvm::constraint::Type::GREATER_THAN:
        return "GREATER_THAN";
        break;
      case tvm::constraint::Type::LOWER_THAN:
        return "LOWER_THAN";
        break;
      case tvm::constraint::Type::DOUBLE_SIDED:
        return "DOUBLE_SIDED";
        break;
    }
  };

  // std::cout << "Processing constraints" << std::endl;
  for(const auto & c : constraints)
  {
    // std::cout << "New constraint, type " << typelambda(c.constraint->type()) << ", jacobian size "
    //           << c.constraint->tSize() << std::endl;
    // If the constraint is used for the substitutions, we skip it.
    if(subs.uses(c.constraint))
      continue;
    abilities_.check(c.constraint, c.requirements); // FIXME: should be done in a parent class
    // XXX c.constraint is a first order provider from which we pull the variables
    // It is a different instance than the function ! and so the variables are not the same directly
    // for(const auto & xi :
    //     static_cast<tvm::constraint::internal::LinearizedTaskConstraint *>(c.constraint.get())->functionVariables())
    // {
    //   std::cout << "function variable " << xi->name() << " ptr: " << xi << std::endl;
    // }
    for(const auto & xi : c.constraint->variables())
    {
      // std::cout << "constraint variable " << xi->name() << " ptr: " << xi << std::endl;
      memory->addVariable(subs.substitute(xi));
    }

    int p = c.requirements->priorityLevel().value();
    // FIXME this call can throw if a variable has been removed or added in the function, as checks call
    // the constraint's variables, need to be sure that they are synced (or just call function variables?)
    if(canBeUsedAsBound(c.constraint, subs, constraint::Type::DOUBLE_SIDED) && p == 0)
      bounds.push_back(c);
    else
    {
      if(p == 0)
      {
        if(c.constraint->isEquality())
          nEq += solver.constraintSize(*c.constraint);
        else
          nIneq += solver.constraintSize(*c.constraint);
      }
      else
      {
        if(c.constraint->type() != constraint::Type::EQUAL)
          throw std::runtime_error("This scheme do not handle inequality constraints with priority level > 0.");
        nObj += c.constraint->size(); // note: we cannot have double sided constraints at this level.
        maxp = std::max(maxp, p);
      }
      constr.push_back(c);
    }
  }
  // we need to add the additional constraints due to the substitutions.
  // They are added at level 0
  for(const auto & c : subs.additionalConstraints())
  {
    nEq += c->size();
    constr.push_back(
        {c, std::make_shared<requirements::SolvingRequirementsWithCallbacks>(requirements::PriorityLevel(0)), false});
  }

  bool autoMinNorm = false;
  if(nObj == 0 && options_.autoDamping().value())
  {
    nObj = memory->variables().totalSize();
    autoMinNorm = true;
  }

  // configure assignments. FIXME: can we find a better way ?
  Assignment::big_ = big_number_;

  // allocating memory for the solver
  solver.startBuild(memory->variables(), nObj, nEq, nIneq, bounds.size() > 0, &subs);
  // memory->assignments.reserve(constr.size() + bounds.size()); //TODO something equivalent

  memory->maxp = maxp;

  // assignments for general constraints
  for(const auto & c : constr)
  {
    int p = c.requirements->priorityLevel().value();
    if(p == 0)
    {
      // TODO check requirements
      solver.addConstraint(c.constraint);
    }
    else
    {
      solver.addObjective(c.constraint, c.requirements, std::pow(*options_.scalarizationWeight(), maxp - p));
    }
  }

  if(autoMinNorm)
    solver.setMinimumNorm();

  // assignments for bounds
  for(const auto & b : bounds)
  {
    int p = b.requirements->priorityLevel().value();
    if(p == 0)
    {
      // TODO check requirements
      solver.addBound(b.constraint);
    }
    else
    {
      throw std::runtime_error("This scheme do not handle inequality constraints with priority level > 0.");
    }
  }

  solver.finalizeBuild();
}

void WeightedLeastSquares::addTask(const LinearizedControlProblem & problem,
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
  std::cout << "addTask: " << &optc->get() << std::endl;

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
    if(p == 0)
    {
      se.addConstraint(c.constraint);
    }
    else
    {
      // We don't adapt maxp, meaning that a constraint with priority level p>max_p will get a weight<1
      se.addObjective({c.constraint, c.requirements, std::pow(*options_.scalarizationWeight(), memory->maxp - p)});
    }
  }
}

void WeightedLeastSquares::removeTask(const LinearizedControlProblem & problem,
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
    if(p == 0)
    {
      se.removeConstraint(c.constraint);
    }
    else
    {
      se.removeObjective(c.constraint);
    }
  }
  memory->removeConstraint(task);
}

WeightedLeastSquares::Memory::Memory(int solverId, std::unique_ptr<solver::abstract::LeastSquareSolver> solver)
: LinearizedProblemComputationData(solverId), solver(std::move(solver))
{}

void WeightedLeastSquares::Memory::reset(std::unique_ptr<solver::abstract::LeastSquareSolver> solver)
{
  LinearizedProblemComputationData::reset();
  this->solver = std::move(solver);
  maxp = 0;
}

void WeightedLeastSquares::Memory::setVariablesToSolution_(tvm::internal::VariableCountingVector & x)
{
  x.set(solver->result());
}

} // namespace scheme

} // namespace tvm
