#pragma once

#include <memory>

#include <Eigen/Core>

#include <tvm/api.h>
#include "ConstraintEnums.h"
#include "defs.h"

namespace tvm
{
  struct Range;
  class Requirements;
  class Variable;
  class VariableVector;

  typedef std::shared_ptr<Eigen::MatrixXd> MatrixPtr;
  typedef std::shared_ptr<Eigen::VectorXd> VectorPtr;
  typedef std::shared_ptr<Range> RangePtr;

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
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr b, ConstraintType ct, RHSType rt);
    /** l <= Ax <= u */
    AssignmentTarget(RangePtr range, MatrixPtr A, VectorPtr l, VectorPtr u, RHSType rt);


    ConstraintType constraintType() const;
    RHSType rhsType() const;

    /** Return the (range.dim x colDim) block of A starting at
    *(range.start,colStart) */
    MatrixRef getA(int colStart, int colDim) const;
    /** Return the segment of l defined by range. */
    VectorRef getl() const;
    /** Return the segment of u defined by range. */
    VectorRef getu() const;
    /** Return the segment of b defined by range. */
    VectorRef getb() const;

    /** Same as getA, and getb, but return only the first or second half of 
      * the row range. This is necessary when double-sided constraints are
      * assigned to matrix/vector with single-sided convention
      */
    MatrixRef getAFirstHalf(int colStart, int colDim) const;
    MatrixRef getASecondHalf(int colStart, int colDim) const;
    VectorRef getbFirstHalf() const;
    VectorRef getbSecondHalf() const;

  private:
    /** Constraint type convention*/
    ConstraintType cstrType_;
    /** RHS type convention*/
    RHSType rhsType_;
    /** Pointer to the row range*/
    RangePtr range_;
    /** Pointers to the target matrix and vectors (when applicable) */
    MatrixPtr A_;
    VectorPtr l_;
    VectorPtr u_;
    VectorPtr b_;
  };
}