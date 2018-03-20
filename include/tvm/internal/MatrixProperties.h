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

#include <utility>

namespace tvm
{

namespace internal
{

  /** This class describes some mathematical properties of a matrix. */
  class TVM_DLLAPI MatrixProperties
  {
  public:
    /** Shape of a matrix*/
    enum Shape
    {
      GENERAL,              //general shape. This includes all other shapes.
      LOWER_TRIANGULAR,     //lower triangular matrix. This includes diagonal matrices.
      UPPER_TRIANGULAR,     //upper triangular matrix. This includes diagonal matrices.
      DIAGONAL,             //diagonal matrices. This includes multiple of the identity matrices (including zero matrix)
      MULTIPLE_OF_IDENTITY, //a*I, where a is a real number. This includes the case a=-1, a=0, and a=1.
      IDENTITY,             //identity matrix I
      MINUS_IDENTITY,       //-I
      ZERO                  //zero matrix
    };

    /** Positiveness property of the matrix. Any options other than NA implies
    * that the matrix is symmetric
    */
    enum Positiveness
    {
      NA,                     // not applicable (matrix is not symmetric) / unknown
      POSITIVE_SEMIDEFINITE,  // all eigenvalues are >=0
      POSITIVE_DEFINITE,      // all eigenvalues are >0
      NEGATIVE_SEMIDEFINITE,  // all eigenvalues are <=0
      NEGATIVE_DEFINITE,      // all eigenvalues are <0
      INDEFINITE,             // eigenvalues are a mix of positive, negative and 0
      NON_ZERO_INDEFINITE,    // eigenvalues are a mix of positive, negative but not 0
    };

    /** A wrapper over a boolean representing the constness of a matrix.*/
    struct Constness
    {
    public:
      Constness(bool b = false) : b_(b) {}
      operator bool() const { return b_; }
    private:
      bool b_;
    };

    /** A wrapper over a boolean representing the invertibility of a matrix.*/
    struct Invertibility
    {
    public:
      Invertibility(bool b = false) : b_(b) {}
      operator bool() const { return b_; }
    private:
      bool b_;
    };

    /** The data given to the constructors may be redundant. For example an
      * identity matrix is constant, invertible and positive definite. The
      * constructors are deducing automatically all what they can from the
      * shape and the positiveness.
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
      * - if a matrix is triangular and symmetric, then it is diagonal.
      */
    MatrixProperties();
    template<typename ... Args>
    MatrixProperties(Args && ... args);

    Shape shape() const;
    Positiveness positiveness() const;
    Constness constness() const;
    Invertibility invertibility() const;

    bool isConstant() const;
    bool isInvertible() const;
    bool isTriangular() const;
    bool isLowerTriangular() const;
    bool isUpperTriangular() const;
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
    bool isIndefinite() const;
    bool isNonZeroIndefinite() const;

  private:
    /** This macro adds a member of type T named \a member to a struct, and
      * a method \a processArgs to process it when it appears in a list of
      * arguments.
      * processArgs return a pair of boolean that is the memberwise "or" of
      * another pair it gets by recursive call and {b1, b2}
      */
#define ADD_ARGUMENT(T, member, b1, b2) \
    T member = T(); \
    template<typename ... Args> \
    std::pair<bool,bool> processArgs(const T& m, const Args& ... args) \
    { \
      static_assert(!check_args<T, Args...>(), \
                    #T" has already been specified"); \
      member = m; \
      auto p = processArgs(args...); \
      return {p.first || b1, p.second || b2}; \
    }

    struct Arguments
    {
      ADD_ARGUMENT(Shape, shape, false, false)
      ADD_ARGUMENT(Positiveness, positiveness, false, false)
      ADD_ARGUMENT(Constness, constant, true, false)
      ADD_ARGUMENT(Invertibility, invertible, false, true)

      std::pair<bool, bool> processArgs() { return{ false, false }; }

    private:
      template<typename T, typename Arg0, typename ... Args>
      static constexpr bool check_args()
      {
        return std::is_same<T, Arg0>::value || check_args<T, Args...>();
      }

      template<typename T>
      static constexpr bool check_args()
      {
        return false;
      }
    };

    void build(const Arguments& args, const std::pair<bool, bool>& checks);

    bool        constant_;
    bool        invertible_;
    Shape       shape_;
    bool        symmetric_;
    Positiveness positiveness_;
  };


  // operators
  TVM_DLLAPI MatrixProperties operator-(const MatrixProperties&);
  TVM_DLLAPI MatrixProperties operator*(double, const MatrixProperties&);
  TVM_DLLAPI MatrixProperties operator+(const MatrixProperties&, const MatrixProperties&);
  TVM_DLLAPI MatrixProperties operator-(const MatrixProperties&, const MatrixProperties&);
  TVM_DLLAPI MatrixProperties operator*(const MatrixProperties&, const MatrixProperties&);

  template<typename ... Args>
  inline MatrixProperties::MatrixProperties(Args && ... args)
  {
    Arguments a;
    auto p = a.processArgs(args...);
    build(a, p);
  }


  inline MatrixProperties::Shape MatrixProperties::shape() const
  {
    return shape_;
  }

  inline MatrixProperties::Positiveness MatrixProperties::positiveness() const
  {
    return positiveness_;
  }

  inline MatrixProperties::Constness MatrixProperties::constness() const
  {
    return constant_;
  }

  inline MatrixProperties::Invertibility MatrixProperties::invertibility() const
  {
    return invertible_;
  }

  inline bool MatrixProperties::isConstant() const
  {
    return constant_;
  }

  inline bool MatrixProperties::isInvertible() const
  {
    return invertible_;
  }

  inline bool MatrixProperties::isTriangular() const
  {
    return shape_ >= Shape::LOWER_TRIANGULAR;
  }

  inline bool MatrixProperties::isLowerTriangular() const
  {
    return shape_ == Shape::LOWER_TRIANGULAR || isDiagonal();
  }

  inline bool MatrixProperties::isUpperTriangular() const
  {
    return shape_ >= Shape::UPPER_TRIANGULAR;
  }

  inline bool MatrixProperties::isDiagonal() const
  {
    return shape_ >= Shape::DIAGONAL;
  }

  inline bool MatrixProperties::isMultipleOfIdentity() const
  {
    return shape_ >= Shape::MULTIPLE_OF_IDENTITY;
  }

  inline bool MatrixProperties::isIdentity() const
  {
    return shape_ == Shape::IDENTITY;
  }

  inline bool MatrixProperties::isMinusIdentity() const
  {
    return shape_ == Shape::MINUS_IDENTITY;
  }

  inline bool MatrixProperties::isZero() const
  {
    return shape_ == Shape::ZERO;
  }

  inline bool MatrixProperties::isSymmetric() const
  {
    return symmetric_;
  }

  inline bool MatrixProperties::isPositiveSemiDefinite() const
  {
    return positiveness_ == Positiveness::POSITIVE_SEMIDEFINITE
        || isPositiveDefinite()
        || isZero();
  }

  inline bool MatrixProperties::isPositiveDefinite() const
  {
    return positiveness_ == Positiveness::POSITIVE_DEFINITE;
  }

  inline bool MatrixProperties::isNegativeSemidefinite() const
  {
    return positiveness_ == Positiveness::NEGATIVE_SEMIDEFINITE
        || isNegativeDefinite()
        || isZero();
  }

  inline bool MatrixProperties::isNegativeDefinite() const
  {
    return positiveness_ == Positiveness::NEGATIVE_DEFINITE;
  }

  inline bool MatrixProperties::isIndefinite() const
  {
    return positiveness_ != Positiveness::NA;
  }

  inline bool MatrixProperties::isNonZeroIndefinite() const
  {
    return positiveness_ == Positiveness::NON_ZERO_INDEFINITE
        || isPositiveDefinite()
        || isNegativeDefinite();
  }

}  // namespace internal

}  // namespace tvm
