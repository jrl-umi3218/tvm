/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

#include <tvm/solver/abstract/LeastSquareSolver.h>
#include <tvm/VariableVector.h>

#include <iostream>


namespace
{
  using namespace tvm;
  using namespace tvm::solver::abstract;
  /** Helper class relying on RAII to update automatically the map from a constraint to
    * the corresponding assignments
    */
  class AutoMap
  {
  public:
    /** Upon destruction of this object, pointers to all new elements in \a observed since
      * calling this constructor will be added to \a target[cstr.get()].
      */
    AutoMap(LinearConstraintPtr cstr, LeastSquareSolver::AssignmentVector& observed, LeastSquareSolver::MapToAssignment& target)
      : observedSize_(observed.size()), observed_(observed), target_(target[cstr.get()]) {}
    ~AutoMap()
    {
      for (size_t i = observedSize_; i < observed_.size(); ++i)
        target_.push_back(&observed_[i]);
    }

  private:
    size_t observedSize_;
    LeastSquareSolver::AssignmentVector& observed_;
    LeastSquareSolver::AssignmentPtrVector& target_;
  };
}

namespace tvm
{

namespace solver
{

namespace abstract
{
  LeastSquareSolver::LeastSquareSolver(bool verbose)
    : objSize_(-1)
    , eqSize_(-1)
    , ineqSize_(-1)
    , buildInProgress_(false)
    , subs_(nullptr)
    , verbose_(verbose)
    , variables_(nullptr)
  {
  }

  void LeastSquareSolver::startBuild(const VariableVector& x, int nObj, int nEq, int nIneq, bool useBounds, const hint::internal::Substitutions& subs)
  {
    assert(nObj >= 0);
    assert(nEq >= 0);
    assert(nIneq >= 0);

    buildInProgress_ = true;
    variables_ = &x;
    first_.clear();
    for (const auto& xi : variables())
    {
      first_[xi.get()] = true;
    }

    subs_ = &subs;

    initializeBuild_(nObj, nEq, nIneq, useBounds);
    nEq_ = nEq;
    nIneq_ = nIneq;
    nObj_ = nObj;
    objSize_ = 0;
    eqSize_ = 0;
    ineqSize_ = 0;
  }

  void LeastSquareSolver::finalizeBuild()
  {
    assert(nObj_ == objSize_);
    assert(nEq_ == eqSize_);
    assert(nIneq_ == ineqSize_);
    buildInProgress_ = false;
  }

  void LeastSquareSolver::addBound(LinearConstraintPtr bound)
  {
    assert(buildInProgress_ && "Attempting to add a bound without calling startBuild first");
    assert(bound->variables().numberOfVariables() == 1 && "A bound constraint can be only on one variable.");
    const auto& xi = bound->variables()[0];
    RangePtr range = std::make_shared<Range>(xi->getMappingIn(variables())); //FIXME: for now we do not keep a pointer on the range nor the target.

    AutoMap autoMap(bound, assignments_, boundToAssigments_);
    bool& first = first_[xi.get()];
    addBound_(bound, range, first);
    first = false;
  }

  void LeastSquareSolver::addConstraint(LinearConstraintPtr cstr)
  {
    assert(buildInProgress_ && "Attempting to add a constraint without calling startBuild first");

    AutoMap autoMap(cstr, assignments_, constraintToAssigments_);
    if (cstr->isEquality())
    {
      addEqualityConstraint_(cstr);
      eqSize_ += constraintSize(cstr);
    }
    else
    {
      addIneqalityConstraint_(cstr);
      ineqSize_ += constraintSize(cstr);
    }
  }

  void LeastSquareSolver::addObjective(LinearConstraintPtr obj, const SolvingRequirementsPtr req, double additionalWeight)
  {
    assert(req->priorityLevel().value() != 0);
    assert(buildInProgress_ && "Attempting to add an objective without calling startBuild first");
    
    if (req->violationEvaluation().value() != requirements::ViolationEvaluationType::L2)
    {
      throw std::runtime_error("[LeastSquareSolver::addObjective]: least-squares only support L2 norm for violation evaluation");
    }
    addObjective_(obj, req, additionalWeight);
    objSize_ += obj->size();
  }


  void LeastSquareSolver::setMinimumNorm()
  {
    assert(nObj_ == variables().totalSize());
    if (!buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to add an objective without calling startBuild first");
    }
    setMinimumNorm_();
  }

  bool LeastSquareSolver::solve()
  {
    if (buildInProgress_)
    {
      throw std::runtime_error("[LeastSquareSolver]: attempting to solve while in build mode");
    }

    preAssignmentProcess_();
    for (auto& a : assignments_)
      a.run();
    postAssignmentProcess_();

    if (verbose_)
      printProblemData_();

    bool b = solve_();

    if (verbose_ || !b)
    {
      printDiagnostic_();
      if (verbose_)
      {
        std::cout << "[LeastSquareSolver::solve] solution: " << result_().transpose() << std::endl;
      }
    }

    return b;
  }

  const Eigen::VectorXd& LeastSquareSolver::result() const
  {
    return result_();
  }

  int LeastSquareSolver::constraintSize(const LinearConstraintPtr& c) const
  {
    if (c->type() != constraint::Type::DOUBLE_SIDED || handleDoubleSidedConstraint_())
    {
      return c->size();
    }
    else
    {
      return 2 * c->size();
    }
  }
}

}

}