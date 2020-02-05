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

#include <tvm/solver/abstract/LeastSquareSolver.h>

#include <eigen-qld/QLD.h>

#include <Eigen/QR>

namespace tvm
{

namespace solver
{
  class QLDLeastSquareConfiguration;

  /** A set of options for QLDLeastSquareSolver */
  class TVM_DLLAPI QLDLeastSquareOptions
  {
    ADD_NON_DEFAULT_OPTION  (big_number,          constant::big_number)
    ADD_NON_DEFAULT_OPTION  (cholesky,            false)
    ADD_NON_DEFAULT_OPTION  (eps,                 1e-6)
    ADD_NON_DEFAULT_OPTION  (verbose,             false)
  public:
    using Config = QLDLeastSquareConfiguration;
  };

  /** An encapsulation of the QLD solver, to solve linear least-squares problems. */
  class TVM_DLLAPI QLDLeastSquareSolver : public abstract::LeastSquareSolver
  {
  public:
    QLDLeastSquareSolver(const QLDLeastSquareOptions & options = {});

  protected:
    void initializeBuild_(int m1, int me, int mi, bool useBounds) override;
    void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) override;
    void addEqualityConstraint_(LinearConstraintPtr cstr) override;
    void addIneqalityConstraint_(LinearConstraintPtr cstr) override;
    void addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight) override;
    void setMinimumNorm_() override;
    void preAssignmentProcess_() override;
    void postAssignmentProcess_() override;
    bool solve_() override;
    virtual const Eigen::VectorXd& result_() const override;
    bool handleDoubleSidedConstraint_() const override { return false; }

    void printProblemData_() const override;
    void printDiagnostic_() const override;

  private:
    using VectorXdTail = decltype(Eigen::VectorXd().tail(1));
    using MatrixXdBottom = decltype(Eigen::MatrixXd().bottomRows(1));

    Eigen::MatrixXd D_;   //We have Q = D^T D
    Eigen::VectorXd e_;   //We have c = D^t e
    Eigen::MatrixXd Q_;
    Eigen::VectorXd c_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    Eigen::VectorXd xl_;
    Eigen::VectorXd xu_;

    MatrixXdBottom Aineq_; //part of A_ corresponding to inequality constraints
    VectorXdTail bineq_; //part of B_ corresponding to inequality constraints

    Eigen::QLD qld_;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_; //TODO add option for ColPiv variant

    bool autoMinNorm_;

    //options
    double big_number_;
    double eps_;
    bool   cholesky_; //compute the Cholesky decomposition before calling the solver.
  };

  /** A factory class to create QLDLeastSquareSolver instances with a given
  * set of options.
  */
  class TVM_DLLAPI QLDLeastSquareConfiguration : public abstract::LeastSquareConfiguration
  {
  public:
    /** Creation of a configuration from a set of options*/
    QLDLeastSquareConfiguration(const QLDLeastSquareOptions & options = {});

    std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

  private:
    QLDLeastSquareOptions options_;
  };

}

}