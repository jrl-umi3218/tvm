#pragma once

/* Copyright 2018-2020 CNRS-UM LIRMM, CNRS-AIST JRL
*
* This file is part of TVM.
*
* TVM is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* TVM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with TVM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <tvm/scheme/abstract/LeastSquareSolver.h>

#include <eigen-lssol/LSSOL_LS.h>

namespace tvm
{

namespace scheme
{

  class TVM_DLLAPI LSSOLLeastSquareSolver : public abstract::LeastSquareSolver
  {
  public:
    LSSOLLeastSquareSolver(double big_number = constant::big_number);

  protected:
    void initializeBuild_(int m1, int me, int mi, bool useBounds) override;
    void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) override;
    void addEqualityConstraint_(LinearConstraintPtr cstr) override;
    void addIneqalityConstraint_(LinearConstraintPtr cstr) override;
    void addObjective_(LinearConstraintPtr cstr, SolvingRequirementsPtr req, double additionalWeight) override;
    bool solve_() override;
    bool handleDoubleSidedConstraint_() const override { return true; }

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

    double big_number_;
  };


  class TVM_DLLAPI LSSOLLeastSquareSolverConfiguration : public abstract::LeastSquareSolverConfiguration
  {
  public:
    LSSOLLeastSquareSolverConfiguration(double big_number = constant::big_number);

    std::unique_ptr<abstract::LeastSquareSolver> createSolver() const override;

  private:
    double big_number_;
  };

}

}