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

#include <eigen-lssol/LSSOL_LS.h>

namespace tvm
{

namespace solver
{
  class TVM_DLLAPI LSSOLLeastSquareOptions
  {
    ADD_NON_DEFAULT_OPTION  (big_number,          constant::big_number)
    ADD_DEFAULT_OPTION      (crashTol,            double)
    ADD_DEFAULT_OPTION      (feasibilityMaxIter,  int)
    ADD_NON_DEFAULT_OPTION  (feasibilityTol,      1e-6)
    ADD_DEFAULT_OPTION      (infiniteBnd,         double)
    ADD_DEFAULT_OPTION      (infiniteStep,        double)
    ADD_DEFAULT_OPTION      (optimalityMaxIter,   int)
    ADD_DEFAULT_OPTION      (persistence,         bool)
    ADD_DEFAULT_OPTION      (printLevel,          int)
    ADD_DEFAULT_OPTION      (rankTol,             double)
    ADD_DEFAULT_OPTION      (type,                Eigen::lssol::eType)
    ADD_NON_DEFAULT_OPTION  (verbose,             false)
    ADD_NON_DEFAULT_OPTION  (warm,                true)
  };



  class TVM_DLLAPI LSSOLLeastSquareSolver : public abstract::LeastSquareSolver
  {
  public:
    LSSOLLeastSquareSolver(const LSSOLLeastSquareOptions& options = {});

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
    bool handleDoubleSidedConstraint_() const override { return true; }

    void printProblemData_() const override;
    void printDiagnostic_() const override;

  private:
    using VectorXdTail = decltype(Eigen::VectorXd().tail(1));

    Eigen::MatrixXd A_;
    Eigen::MatrixXd C_;
    Eigen::VectorXd b_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    VectorXdTail    cl_;   // part of l_ corresponding to general constraints
    VectorXdTail    cu_;   // part of u_ corresponding to general constraints

    Eigen::LSSOL_LS ls_;

    bool autoMinNorm_;
    double big_number_;
  };


  class TVM_DLLAPI LSSOLLeastSquareConfiguration : public abstract::LeastSquareConfiguration
  {
  public:
    LSSOLLeastSquareConfiguration(const LSSOLLeastSquareOptions& options = {});

    std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

  private:
    LSSOLLeastSquareOptions options_;
  };

}

}