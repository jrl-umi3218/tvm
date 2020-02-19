/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include <tvm/scheme/WeightedLeastSquares.h>

#include <tvm/LinearizedControlProblem.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/utils/internal/map.h>

#include <iostream>


namespace tvm
{

namespace scheme
{
  using namespace internal;
  using VET = requirements::ViolationEvaluationType;

  const internal::SchemeAbilities WeightedLeastSquares::abilities_ = { 2, {{0, {true, {VET::L2}}}, {1,{false, {VET::L2}}}}, true };

  bool WeightedLeastSquares::solve_(LinearizedControlProblem& problem, internal::ProblemComputationData* data) const
  {
    Memory* memory = dynamic_cast<Memory*>(data);
    return memory->solver->solve();
  }

  std::unique_ptr<WeightedLeastSquares::Memory> WeightedLeastSquares::createComputationData_(const LinearizedControlProblem& problem) const
  {
    auto memory = std::unique_ptr<Memory>(new Memory(id(), solverFactory_->createSolver()));
    auto& solver = *memory->solver;

    const auto& constraints = problem.constraints();
    const auto& subs = problem.substitutions();

    std::vector<LinearConstraintWithRequirements> constr;
    std::vector<LinearConstraintWithRequirements> bounds;

    //scanning constraints
    int nEq = 0;
    int nIneq = 0;
    int nObj = 0;
    int maxp = 0;
    for (const auto& c : constraints)
    {
      // If the constraint is used for the substitutions, we skip it.
      if (subs.uses(c.constraint))
        continue;
      abilities_.check(c.constraint, c.requirements); //FIXME: should be done in a parent class
      for (const auto& xi: c.constraint->variables())
        memory->addVariable(subs.substitute(xi));

      int p = c.requirements->priorityLevel().value();
      if (canBeUsedAsBound(c.constraint, subs, constraint::Type::DOUBLE_SIDED)
          && p == 0)
        bounds.push_back(c);
      else
      {
        if (p == 0)
        {
          if (c.constraint->isEquality())
            nEq += solver.constraintSize(c.constraint);
          else
            nIneq += solver.constraintSize(c.constraint);
        }
        else
        {
          if (c.constraint->type() != constraint::Type::EQUAL)
            throw std::runtime_error("This scheme do not handle inequality constraints with priority level > 0.");
          nObj += c.constraint->size();  //note: we cannot have double sided constraints at this level.
          maxp = std::max(maxp, p);
        }
        constr.push_back(c);
      }
    }
    // we need to add the additional constraints due to the substitutions.
    // They are added at level 0
    for (const auto& c : subs.additionalConstraints())
    {
      nEq += c->size();
      constr.push_back({ c, std::make_shared<requirements::SolvingRequirements>(requirements::PriorityLevel(0)), false });
    }

    bool autoMinNorm = false;
    if (nObj == 0 && options_.autoDamping().value())
    {
      nObj = memory->variables().totalSize();
      autoMinNorm = true;
    }

    // configure assignments. FIXME: can we find a better way ?
    Assignment::big_ = big_number_;

    //allocating memory for the solver
    solver.startBuild(memory->variables(), nObj, nEq, nIneq, bounds.size() > 0, subs);
    //memory->assignments.reserve(constr.size() + bounds.size()); //TODO something equivalent

    //assigments for general constraints
    for (const auto& c : constr)
    {
      int p = c.requirements->priorityLevel().value();
      if (p == 0)
      {
        //TODO check requirements
        solver.addConstraint(c.constraint);
      }
      else
      {
        solver.addObjective(c.constraint, c.requirements, std::pow(*options_.scalarizationWeight(), maxp - p));
      }
    }

    if (autoMinNorm)
      solver.setMinimumNorm();

    //assigments for bounds
    for (const auto& b : bounds)
    {
      const auto& xi = b.constraint->variables()[0];
      int p = b.requirements->priorityLevel().value();
      if (p == 0)
      {
        //TODO check requirements
        solver.addBound(b.constraint);
      }
      else
      {
        throw std::runtime_error("This scheme do not handle inequality constraints with priority level > 0.");
      }
    }

    solver.finalizeBuild();

    return memory;
  }

  WeightedLeastSquares::Memory::Memory(int solverId, std::unique_ptr<solver::abstract::LeastSquareSolver> solver)
    : ProblemComputationData(solverId)
    , solver(std::move(solver))
  {
  }

  void WeightedLeastSquares::Memory::setVariablesToSolution_(VariableVector& x)
  {
    x.value(solver->result());
  }

}  // namespace scheme

}  // namespace tvm
