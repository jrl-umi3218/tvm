#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
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

#include <tvm/constraint/abstract/LinearConstraint.h>

namespace tvm
{

namespace constraint
{

  /** The most basic linear constraint where the matrix and the vector(s)
   * are constant.
    */
  class TVM_DLLAPI BasicLinearConstraint : public abstract::LinearConstraint
  {
  public:
    /** Ax = 0, Ax <= 0 or Ax >= 0. */
    BasicLinearConstraint(const MatrixConstRef& A, VariablePtr x, Type ct);

    /** A_{i}x_{i} = 0, A_{i}x_{i} <= 0 or A_{i}x_{i} >= 0 */
    BasicLinearConstraint(const std::vector<MatrixConstRef>& A,
                          const std::vector<VariablePtr>& x,
                          Type ct);

    /** Ax = +/-b, Ax <= +/-b or Ax >= +/-b */
    BasicLinearConstraint(const MatrixConstRef& A,
                          VariablePtr x,
                          const VectorConstRef& b,
                          Type ct, RHS cr = RHS::AS_GIVEN);

    /** A_{i}x_{i} = +/-b, A_{i}x_{i} <= +/-b or A_{i}x_{i} >= +/-b */
    BasicLinearConstraint(const std::vector<MatrixConstRef>& A,
                          const std::vector<VariablePtr>& x,
                          const VectorConstRef& b,
                          Type ct, RHS cr = RHS::AS_GIVEN);

    /** l <= Ax <= u */
    BasicLinearConstraint(const MatrixConstRef& A,
                          VariablePtr x,
                          const VectorConstRef& l,
                          const VectorConstRef& u,
                          RHS cr = RHS::AS_GIVEN);

    /** l <= A_{i}x_{i} <= u */
    BasicLinearConstraint(const std::vector<MatrixConstRef>& A,
                          const std::vector<VariablePtr>& x,
                          const VectorConstRef& l,
                          const VectorConstRef& u,
                          RHS cr = RHS::AS_GIVEN);

    /** Uninitialized data. Allocate memory for a constraint with \p m rows. */
    BasicLinearConstraint(int m, std::vector<VariablePtr>& x,
                          Type ct, RHS cr = RHS::AS_GIVEN);

    /** Set the matrix A corresponding to variable x.*/
    void A(const MatrixConstRef& A, const Variable& x);
    /** Shortcut for when there is a single variable.*/
    void A(const MatrixConstRef& A);
    /** Set b */
    void b(const VectorConstRef& b);
    using abstract::LinearConstraint::l;
    /** Set l */
    void l(const VectorConstRef& l);
    using abstract::LinearConstraint::u;
    /** Set u */
    void u(const VectorConstRef& u);
  private:
    void add(const Eigen::MatrixXd& A, VariablePtr x);
  };

}  // namespace constraint

}  // namespace tvm
