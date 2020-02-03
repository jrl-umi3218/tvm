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

#pragma once

#include <tvm/solver/LSSOLLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>


namespace tvm
{

namespace solver
{
  LSSOLLeastSquareSolver::LSSOLLeastSquareSolver(bool verbose, double big_number)
    : LeastSquareSolver(verbose)
    , cl_(l_.tail(0))
    , cu_(u_.tail(0))
    , big_number_(big_number)
    , autoMinNorm_(false)
  {
  }

  void LSSOLLeastSquareSolver::initializeBuild_(int m1, int me, int mi, bool useBounds)
  {
    int n = variables().totalSize();
    int m0 = me + mi;
    A_.resize(m1, n);
    A_.setZero();
    C_.resize(m0, n);
    C_.setZero();
    b_.resize(m1);
    b_.setZero();
    l_ = Eigen::VectorXd::Constant(m0 + n, -big_number_);
    u_ = Eigen::VectorXd::Constant(m0 + n, +big_number_);
    new(&cl_) VectorXdTail(l_.tail(m0));
    new(&cu_) VectorXdTail(u_.tail(m0));
    ls_.resize(n, m0, Eigen::lssol::eType::LS1);
    // TODO: move to options
    ls_.warm(true);
    ls_.feasibilityTol(1e-6);

    autoMinNorm_ = false;
  }

  void LSSOLLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
  {
    scheme::internal::AssignmentTarget target(range, l_, u_);
    addAssignement(bound, target, bound->variables()[0], first);
  }

  void LSSOLLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(eqSize_+ineqSize_, cstr->size());
    scheme::internal::AssignmentTarget target(r, C_, cl_, cu_, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, nullptr, target, variables(), *substitutions());
  }

  void LSSOLLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(eqSize_ + ineqSize_, cstr->size());
    scheme::internal::AssignmentTarget target(r, C_, cl_, cu_, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, nullptr, target, variables(), *substitutions());
  }

  void LSSOLLeastSquareSolver::addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight)
  {
    RangePtr r = std::make_shared<Range>(objSize_, cstr->size());
    scheme::internal::AssignmentTarget target(r, A_, b_, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, req, target, variables(), *substitutions(), additionalWeight);
  }

  void LSSOLLeastSquareSolver::setMinimumNorm_()
  {
    autoMinNorm_ = true;
    b_.setZero();
  }

  void LSSOLLeastSquareSolver::preAssignmentProcess_()
  {
    // LSSOL is overwritting A during the resolution.
    // We need to make sure that A is clean before assignments are carried out.
    if (!autoMinNorm_)
      A_.setZero();
  }

  void LSSOLLeastSquareSolver::postAssignmentProcess_()
  {
    if (autoMinNorm_)
      A_.setIdentity();
  }

  bool LSSOLLeastSquareSolver::solve_()
  {
    return ls_.solve(A_, b_, C_, l_, u_);
  }

  const Eigen::VectorXd& LSSOLLeastSquareSolver::result_() const
  {
    return ls_.result();
  }

  void LSSOLLeastSquareSolver::printProblemData_() const
  {
    std::cout << "A =\n" << A_ << std::endl;
    std::cout << "b = " << b_.transpose() << std::endl;
    std::cout << "C =\n" << C_ << std::endl;
    std::cout << "l = " << l_.transpose() << std::endl;
    std::cout << "u = " << u_.transpose() << std::endl;
  }

  void LSSOLLeastSquareSolver::printDiagnostic_() const
  {
    std::cout << ls_.inform() << std::endl;
    ls_.print_inform();
  }


  LSSOLLeastSquareSolverConfiguration::LSSOLLeastSquareSolverConfiguration(bool verbose, double big_number)
    : LeastSquareSolverConfiguration("lssol")
    , big_number_(big_number)
    , verbose_(verbose)
  {
  }
  
  std::unique_ptr<abstract::LeastSquareSolver> LSSOLLeastSquareSolverConfiguration::createSolver() const
  {
    return std::make_unique<LSSOLLeastSquareSolver>(big_number_);
  }
}

}