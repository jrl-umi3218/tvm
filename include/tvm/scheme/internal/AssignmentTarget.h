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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/constraint/enums.h>

#include <Eigen/Core>

#include <memory>

namespace tvm
{

namespace scheme
{

namespace internal
{

  enum class TargetType
  {
    Linear,
    Quadratic
  };

  /** This class describes the matrix and vector(s) rows in which a given
   * constraint needs to be copied, and the convention to be used for those
   * matrix and vectors.
   *
   * If the target is quadratic form, the whole matrix and vector are
   * returned.
   *
   * Note that you don't necessarily need to allocate the matrices and
   * vectors dynamically. You can use use shared_ptr with aliasing as well.
   *
   * FIXME we are using pointers here so that the ResolutionScheme can
   * update the values (typically change the range or resize the matrices).
   * A safer alternative would be to have an event management here.
   */
  class TVM_DLLAPI AssignmentTarget
  {
  public:
    /** Ax = 0, Ax <= 0 or Ax >= 0. */
    AssignmentTarget(RangePtr range, MatrixPtr A, constraint::Type ct);
    /** Ax = +/-b, Ax <= +/-b or Ax >= +/-b
      *
      * \param shift shifts the range for the vectors, i.e. the target is the rows
      * starting at range.start for A, but range.start+shift for b (and l and u).
      */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr b, constraint::Type ct, constraint::RHS cr, int shift=0);

    /** l <= Ax <= u
      *
      * \param shift shifts the range for the vectors, i.e. the target is the rows
      * starting at range.start for A, but range.start+shift for b (and l and u).
      */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr l, VectorPtr u, constraint::RHS cr, int shift=0);

    /** Quadratic function 1/2 x^T Q x +\epsilon q, where \epsilon = 0, 1 or -1 depending on cr.*/
    AssignmentTarget(MatrixPtr Q, VectorPtr q, constraint::RHS cr);


    TargetType targetType() const;
    constraint::Type constraintType() const;
    constraint::RHS constraintRhs() const;

    /** Return the (range.dim x colDim) block of A starting at
    *(range.start,colStart) */
    MatrixRef A(int colStart, int colDim) const;
    /** Return the whole quadratic matrix*/
    MatrixRef Q() const;
    /** Return the segment of l defined by range. */
    VectorRef l() const;
    /** Return the segment of u defined by range. */
    VectorRef u() const;
    /** Return the segment of b defined by range. */
    VectorRef b() const;
    /** Return the whole vector q.*/
    VectorRef q() const;

    /** Same as A(...), and b(), but return only the first or second half of
      * the row range. This is necessary when double-sided constraints are
      * assigned to matrix/vector with single-sided convention
      */
    MatrixRef AFirstHalf(int colStart, int colDim) const;
    MatrixRef ASecondHalf(int colStart, int colDim) const;
    VectorRef bFirstHalf() const;
    VectorRef bSecondHalf() const;

  private:
    /** Type of target*/
    TargetType targetType_;
    /** Constraint type convention*/
    constraint::Type cstrType_;
    /** RHS type convention*/
    constraint::RHS constraintRhs_;
    /** Pointer to the row range*/
    RangePtr range_;
    /** Shift or range for the vectors*/
    int shift_;
    /** Pointers to the target matrix and vectors (when applicable) */
    MatrixPtr A_;
    MatrixPtr Q_;
    VectorPtr l_;
    VectorPtr u_;
    VectorPtr b_;
    VectorPtr q_;
  };

}  // namespace internal

}  // namespace scheme

}  // namespace tvm
