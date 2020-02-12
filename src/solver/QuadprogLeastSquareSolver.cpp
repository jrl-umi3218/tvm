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

#include <tvm/solver/QuadprogLeastSquareSolver.h>

#include <tvm/scheme/internal/AssignmentTarget.h>

#include <iostream>


namespace tvm
{

namespace solver
{
  QuadprogLeastSquareSolver::QuadprogLeastSquareSolver(const QuadprogLeastSquareOptions& options)
    : LeastSquareSolver(options.verbose().value())
    , Aineq_(A_.middleRows(0,0))
    , bineq_(b_.segment(0, 0))
    , xl_(b_.segment(0, 0))
    , xu_(b_.segment(0,0))
    , big_number_(options.big_number().value())
    , cholesky_(options.cholesky().value())
    , choleskyDamping_(options.choleskyDamping().value())
    , damping_(options.damping().value())
    , autoMinNorm_(false)
  {
  }

  void QuadprogLeastSquareSolver::initializeBuild_(int m1, int me, int mi, bool useBounds)
  {
    int n = variables().totalSize();
    int m0 = me + mi;
    underspecifiedObj_ = m1 < n;
    if (cholesky_ && underspecifiedObj_)
    {
      D_.resize(m1 + n, n);
      D_.bottomRows(n).setZero();
      D_.bottomRows(n).diagonal().setConstant(choleskyDamping_);
    }
    else
      D_.resize(m1, n);
    e_.resize(m1);
    if (!cholesky_)
      Q_.resize(n, n);
    c_.resize(n);
    if (useBounds)
    {
      mib_ = mi + 2 * n;
      A_.resize(m0+2*n, n);
      b_.resize(m0+2*n);
      A_.middleRows(m0, n).setIdentity();
      A_.middleRows(m0, n).diagonal() *= -1;
      A_.bottomRows(n).setIdentity();
      new(&xl_) VectorXdSeg(b_.segment(m0, n));
      new(&xu_) VectorXdSeg(b_.segment(m0 + n, n));
      xl_.setConstant(-big_number_);
      xu_.setConstant(+big_number_);
    }
    else
    {
      mib_ = mi;
      A_.resize(m0, n);
      b_.resize(m0);
    }
    new(&Aineq_) MatrixXdRows(A_.middleRows(me, mi));
    new(&bineq_) VectorXdSeg(b_.tail(mi));
    if (useBounds)
      qpd_.problem(n, me, mi + 2 * n);
    else
      qpd_.problem(n, me, mi);
    if (cholesky_)
    {
      if (underspecifiedObj_)
        new(&qr_) Eigen::HouseholderQR<Eigen::MatrixXd>(m1 + n, n);
      else
        new(&qr_) Eigen::HouseholderQR<Eigen::MatrixXd>(m1, n);
    }

    autoMinNorm_ = false;
  }

  void QuadprogLeastSquareSolver::addBound_(LinearConstraintPtr bound, RangePtr range, bool first)
  {
    // Here, we fill xl_ as if it is would be used in the solver as a lower bound, because
    // for now there is no path in Assignment allowing for bounds of the form -x <= -xl
    // TODO: extend Assignment for that.
    scheme::internal::AssignmentTarget target(range, xl_, xu_);
    addAssignement(bound, target, bound->variables()[0], first);
  }

  void QuadprogLeastSquareSolver::addEqualityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(eqSize_, cstr->size());
    scheme::internal::AssignmentTarget target(r, A_, b_, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, nullptr, target, variables(), *substitutions());
  }

  void QuadprogLeastSquareSolver::addIneqalityConstraint_(LinearConstraintPtr cstr)
  {
    RangePtr r = std::make_shared<Range>(ineqSize_, constraintSize(cstr));
    scheme::internal::AssignmentTarget target(r, Aineq_, bineq_, constraint::Type::LOWER_THAN, constraint::RHS::AS_GIVEN);
    addAssignement(cstr, nullptr, target, variables(), *substitutions());
  }

  void QuadprogLeastSquareSolver::addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight)
  {
    RangePtr r = std::make_shared<Range>(objSize_, cstr->size());
    scheme::internal::AssignmentTarget target(r, D_, e_, constraint::Type::EQUAL, constraint::RHS::OPPOSITE);
    addAssignement(cstr, req, target, variables(), *substitutions(), additionalWeight);
  }

  void QuadprogLeastSquareSolver::setMinimumNorm_()
  {
    autoMinNorm_ = true;
    Q_.setIdentity();
    c_.setZero();
  }

  void QuadprogLeastSquareSolver::preAssignmentProcess_()
  {
    //signs on xl will be flipped later so we nned to reinitialize
    xl_.setConstant(-big_number_);
  }

  void QuadprogLeastSquareSolver::postAssignmentProcess_()
  {
    // we need to flip the signs for xl
    xl_ = -xl_;

    if (!autoMinNorm_)
    {
      c_.noalias() = D_.topRows(m1_).transpose() * e_;

      if (cholesky_)
      {
        int n = variables().totalSize();
        qr_.compute(D_);
        // we put R^{-1} in D
        D_.topRows(n).setIdentity();
        qr_.matrixQR().topRows(n).template triangularView<Eigen::Upper>().solveInPlace(D_.topRows(n));
      }
      else
      {
        Q_.noalias() = D_.transpose() * D_;   //TODO check if this can be optimized: Quadprog might need only half the matrix
        Q_.diagonal().array() += damping_;
      }
    }
  }

  bool QuadprogLeastSquareSolver::solve_()
  {
    if (cholesky_ && !autoMinNorm_)
    {
      int n = variables().totalSize();
      return qpd_.solve(D_.topRows(n), c_,
        A_.topRows(me_), b_.topRows(me_),
        A_.bottomRows(mib_), b_.bottomRows(mib_),
        true);
    }
    else
    {
      return qpd_.solve(Q_, c_,
        A_.topRows(me_), b_.topRows(me_),
        A_.bottomRows(mib_), b_.bottomRows(mib_),
        false);
    }
  }

  const Eigen::VectorXd& QuadprogLeastSquareSolver::result_() const
  {
    return qpd_.result();
  }

  void QuadprogLeastSquareSolver::printProblemData_() const
  {
    if (cholesky_)
    {
      int n = variables().totalSize();
      std::cout << "R =\n" << qr_.matrixQR().topRows(n).template triangularView<Eigen::Upper>().toDenseMatrix() << std::endl;
    }
    else
      std::cout << "`Q =\n" << Q_ << std::endl;
    std::cout << "c = " << c_.transpose() << std::endl;
    std::cout << "A =\n" << A_ << std::endl;
    std::cout << "b = " << b_.transpose() << std::endl;
  }

  void QuadprogLeastSquareSolver::printDiagnostic_() const
  {
    std::cout << "Quadprog fail code = " << qpd_.fail() << " (0 is success)" << std::endl;
  }

  QuadprogLeastSquareConfiguration::QuadprogLeastSquareConfiguration(const QuadprogLeastSquareOptions& options)
    : LeastSquareConfiguration("quadprog")
    , options_(options)
  {
  }

  std::unique_ptr<abstract::LeastSquareSolver> QuadprogLeastSquareConfiguration::createSolver() const
  {
    return std::make_unique<QuadprogLeastSquareSolver>(options_);
  }
}

}