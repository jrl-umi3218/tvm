#pragma once

#include <memory>

#include <Eigen/Core>

#include <tvm/api.h>
#include "ConstraintEnums.h"
#include "defs.h"

namespace tvm
{
  class Requirements;

  /** This class describes the matrix and vector(s) rows in which a given
  * constraint needs to be copied, and the convention to be used for those
  * matrix and vectors.
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
    AssignmentTarget(RangePtr range, MatrixPtr A, ConstraintType ct);
    /** Ax = +/-b, Ax <= +/-b or Ax >= +/-b */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr b, ConstraintType ct, ConstraintRHS cr);
    /** l <= Ax <= u */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr l, VectorPtr u, ConstraintRHS cr);


    ConstraintType constraintType() const;
    ConstraintRHS constraintRhs() const;

    /** Return the (range.dim x colDim) block of A starting at
    *(range.start,colStart) */
    MatrixRef A(int colStart, int colDim) const;
    /** Return the segment of l defined by range. */
    VectorRef l() const;
    /** Return the segment of u defined by range. */
    VectorRef u() const;
    /** Return the segment of b defined by range. */
    VectorRef b() const;

    /** Same as A(...), and b(), but return only the first or second half of 
      * the row range. This is necessary when double-sided constraints are
      * assigned to matrix/vector with single-sided convention
      */
    MatrixRef AFirstHalf(int colStart, int colDim) const;
    MatrixRef ASecondHalf(int colStart, int colDim) const;
    VectorRef bFirstHalf() const;
    VectorRef bSecondHalf() const;

  private:
    /** Constraint type convention*/
    ConstraintType cstrType_;
    /** RHS type convention*/
    ConstraintRHS constraintRhs_;
    /** Pointer to the row range*/
    RangePtr range_;
    /** Pointers to the target matrix and vectors (when applicable) */
    MatrixPtr A_;
    VectorPtr l_;
    VectorPtr u_;
    VectorPtr b_;
  };
}