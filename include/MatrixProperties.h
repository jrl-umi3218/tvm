#pragma once

#include "tvm/api.h"

namespace tvm
{
  enum class MatrixShape
  {
    GENERAL,
    DIAGONAL,
    MULTIPLE_OF_IDENTITY,
    IDENTITY,
    MINUS_IDENTITY,
    ZERO
  };

  enum class Positivness
  {
    NA,                     // not applicable (matrix is not symmetric) / unknown
    POSITIVE_SEMIDEFINITE,  // all eigenvalues are >=0
    POSITIVE_DEFINITE,      // all eigenvalues are >0
    NEGATIVE_SEMIDEFINITE,  // all eigenvalues are <=0
    NEGATIVE_DEFINITE,      // all eigenvalues are <0
    UNDEFINITE,             // eigenvalues are a mix of positive, negative and 0
    NON_ZERO_UNDEFINITE,    // eigenvalues are a mix of positive, negative but not 0
  };

  /** This class describes some mathematical properties of a matrix.
    */
  class TVM_DLLAPI MatrixProperties
  {
  public:
    /** The data given to the constructors may be redundant. For example an 
      * identity matrix is constant, invertible and positive definite. The 
      * constructors are deducing automatically all what they can from the 
      * arguments, first from the shape only, then from the shape and 
      * the positivness.
      * The constructors use user-given data when they add information to what
      * they can deduce. If the user-given data are less precise but compatible
      * with what has been deduced, they are discarded. If they are 
      * contradicting the deductions, an assertion is fired.
      *
      * Here are some examples:
      * - a multiple-of-identity matrix can only be said to be symmetric and
      * undefinite. If the user specifies it is positive-semidefinite, this 
      * will be recorded. If additionnally it is specified to be invertible, it
      * will be deduced that the matrix is positive definite.
      * - if a minus-identity matrix is said to be non-zero undefinite, this 
      * caracteristic will be discarded as it can be automatically deduced that
      * the matrix is negative definite. If it is said to be positive definite,
      * non constant or non invertible, an assertion will be fire as this
      * contradicts what can be deduced.
      */
    MatrixProperties(MatrixShape shape = MatrixShape::GENERAL, Positivness positivness = Positivness::NA);
    MatrixProperties(bool constant, MatrixShape shape = MatrixShape::GENERAL, Positivness positivness = Positivness::NA);
    MatrixProperties(bool invertible, bool constant = false, MatrixShape shape = MatrixShape::GENERAL, Positivness positivness = Positivness::NA);

    MatrixShape shape() const;
    Positivness positivness() const;

    bool isConstant() const;
    bool isInvertible() const;
    bool isDiagonal() const;
    bool isMultipleOfIdentity() const;
    bool isIdentity() const;
    bool isMinusIdentity() const;
    bool isZero() const;
    bool isSymmetric() const;
    bool isPositiveSemiDefinite() const;
    bool isPositiveDefinite() const;
    bool isNegativeSemidefinite() const;
    bool isNegativeDefinite() const;
    bool isUndefinite() const;
    bool isNonZeroUndefinite() const;

  private:
    bool        constant_;
    bool        invertible_;
    MatrixShape shape_;
    bool        symmetric_;
    Positivness positivness_;
  };

  inline MatrixShape MatrixProperties::shape() const
  {
    return shape_;
  }

  inline Positivness MatrixProperties::positivness() const
  {
    return positivness_;
  }

  inline bool MatrixProperties::isConstant() const
  {
    return constant_;
  }

  inline bool MatrixProperties::isInvertible() const
  {
    return invertible_;
  }

  inline bool MatrixProperties::isDiagonal() const
  {
    return shape_ == MatrixShape::DIAGONAL || isMultipleOfIdentity();
  }

  inline bool MatrixProperties::isMultipleOfIdentity() const
  {
    return shape_ == MatrixShape::MULTIPLE_OF_IDENTITY 
        || isIdentity()
        || isMinusIdentity()
        || isZero();
  }

  inline bool MatrixProperties::isIdentity() const
  {
    return shape_ == MatrixShape::IDENTITY;
  }

  inline bool MatrixProperties::isMinusIdentity() const
  {
    return shape_ == MatrixShape::MINUS_IDENTITY;
  }

  inline bool MatrixProperties::isZero() const
  {
    return shape_ == MatrixShape::ZERO;
  }

  inline bool MatrixProperties::isSymmetric() const
  {
    return symmetric_;
  }

  inline bool MatrixProperties::isPositiveSemiDefinite() const
  {
    return positivness_ == Positivness::POSITIVE_SEMIDEFINITE 
        || isPositiveDefinite() 
        || isZero();
  }

  inline bool MatrixProperties::isPositiveDefinite() const
  {
    return positivness_ == Positivness::POSITIVE_DEFINITE;
  }

  inline bool MatrixProperties::isNegativeSemidefinite() const
  {
    return positivness_ == Positivness::NEGATIVE_SEMIDEFINITE 
        || isNegativeDefinite()
        || isZero();
  }

  inline bool MatrixProperties::isNegativeDefinite() const
  {
    return positivness_ == Positivness::NEGATIVE_DEFINITE;
  }

  inline bool MatrixProperties::isUndefinite() const
  {
    return positivness_ == Positivness::UNDEFINITE 
        || isNonZeroUndefinite() 
        || isPositiveSemiDefinite()
        || isNegativeSemidefinite();
  }

  inline bool MatrixProperties::isNonZeroUndefinite() const
  {
    return positivness_ == Positivness::NON_ZERO_UNDEFINITE
        || isPositiveDefinite()
        || isNegativeDefinite();
  }

}