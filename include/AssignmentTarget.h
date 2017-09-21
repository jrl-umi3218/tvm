#pragma once

#include <memory>

#include <Eigen/Core>

#include <tvm/api.h>
#include "ConstraintEnums.h"
#include "defs.h"

namespace tvm
{
  class Requirements;

  enum class TargetType
  {
    Linear,
    Quadratic
  };

  /** This class describes the matrix and vector(s) rows in which a given
  * constraint needs to be copied, and the convention to be used for those
  * matrix and vectors.
  * If the target is quadratic form, the whole matrix and vector are returned.
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
    /** Ax = +/-b, Ax <= +/-b or Ax >= +/-b 
      * 
      * shift shifts the range for the vectors, i.e. the target is the rows 
      * starting at range.start for A, but range.start+shift for b (and l and u).
      */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr b, ConstraintType ct, ConstraintRHS cr, int shift=0);
    /** l <= Ax <= u */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr l, VectorPtr u, ConstraintRHS cr, int shift=0);
    /** Quadratic function 1/2 x^T Q x +\epsilon q, where \epsilon = 0, 1 or -1 depending on cr.*/
    AssignmentTarget(MatrixPtr Q, VectorPtr q, ConstraintRHS cr);


    TargetType targetType() const;
    ConstraintType constraintType() const;
    ConstraintRHS constraintRhs() const;

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
    ConstraintType cstrType_;
    /** RHS type convention*/
    ConstraintRHS constraintRhs_;
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
}