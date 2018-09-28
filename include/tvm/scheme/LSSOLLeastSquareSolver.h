#pragma once

/* Copyright 2018 CNRS-UM LIRMM, CNRS-AIST JRL
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

namespace tvm
{

namespace scheme
{

  class TVM_DLLAPI LSSOLLeastSquareSolver : public abstract::LeastSquareSolver
  {
  public:

  protected:
    void initializeBuild_(int m0, int me, int mi, bool useBounds) override;
    void addBound_(LinearConstraintPtr bound, RangePtr range, bool first) override;
    void addEqualityConstraint_() override;
    void addIneqalityConstraint_() override;
    void addObjective_() override;

  private:
    Eigen::MatrixXd A_;
    Eigen::MatrixXd C_;
    Eigen::VectorXd b_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
  };

}

}