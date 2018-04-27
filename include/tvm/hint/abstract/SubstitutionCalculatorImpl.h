#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
#include <tvm/VariableVector.h>
#include <tvm/Range.h>

#include <vector>

namespace tvm
{

namespace hint
{

namespace abstract
{
  /** Given a set of variables aggregated as \p x, and a set of constraints
    * that, once aggregated, writes A x + ..., this base class proposes a set of
    * operations related to A that are useful for performing substitutions.
    * The operations revolve around 3 matrices:
    *  - A^#, a generalize inverse of A (i.e. a matrix such that A A^# A = A)
    *  - N, a basis of the nullspace of A. It does not need to be orthonormal.
    *  - S, a basis of the nullspace of A^T. It does not need to be orthonormal.
    *
    * The choice of A^#, N and S is implemented by derivation of this base class.
    * A generic implementation is given by GenericCalculator::Impl
    */
  class TVM_DLLAPI SubstitutionCalculatorImpl
  {
  public:
    virtual ~SubstitutionCalculatorImpl() = default;
    /** Update the internal computations based on the current value of A, i.e
      * the current values of the constraints' jacobian matrices.
      */
    void update();
    /** If \minus = \false, perform \p outA = A^# * \p in and \p outS = S^T * \p in
      * otherwise perform \p outA = - A^# * \p in and \p outS = S^T * \p in
      */
    void premultiplyByASharpAndSTranspose(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const;
    /** Compute \p out = \p in * \p N. 
      * If \p add is \p true, perform \p out += \p in * \p N instead.
      */
    void postMultiplyByN(MatrixRef out, const MatrixConstRef& in, bool add = false) const;
    /** Compute \p out = \p in * \p N(r,:). 
      * If \p add is \p true, perform \p out += \p in * \p N(r,:) instead.
      */
    void postMultiplyByN(MatrixRef out, const MatrixConstRef& in, Range r, bool add = false) const;
    /** Return N as a dense matrix.*/
    const Eigen::MatrixXd& N() const;

    /** Number of lines of A (i.e. sum of the constraints' sizes.*/
    Eigen::DenseIndex m() const;
    /** Size of x (i.e. sum of the size of the variables to be substituted.*/
    Eigen::DenseIndex n() const;
    /** Rank of A*/
    Eigen::DenseIndex r() const;

  protected:
    /** Constructor
      * \param cstr the list of constraints
      * \param x the list of variables
      * \param rank the rank of A
      */
    SubstitutionCalculatorImpl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank);
    
    /** Specify wheter A is constant (in which case only the first update will
      * actually perform computations.
      */
    void constant(bool c);
    /** Return wheter A is constant*/
    bool constant() const;

    /** Return the matrix A*/
    const Eigen::MatrixXd& A() const;

    /** Return true if there is only one variable and one constraint*/
    bool isSimple() const;
    
    /** Handle for the derived class to perform the computations in update()*/
    virtual void update_() = 0;
    /** Computations for premultiplyByASharpAndSTranspose()*/
    virtual void premultiplyByASharpAndSTranspose_(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const = 0;
    /** Computations for postMultiplyByN()*/
    virtual void postMultiplyByN_(MatrixRef out, const MatrixConstRef& in, bool add) const;
    /** Computations for postMultiplyByN(). By default it uses N()*/
    virtual void postMultiplyByN_(MatrixRef out, const MatrixConstRef& in, Range r, bool add) const;

    /** Copy in A_ the values of the relevant jacobian matrices.*/
    void fillA();

    /** The matrix N*/
    Eigen::MatrixXd N_;
    /** The list of constraints.*/
    std::vector<LinearConstraintPtr> constraints_;
    /** The list of variables.*/
    VariableVector variables_;

  private:
    /** An association between a pair (variable, constraint) and a matrix block.*/
    struct FillData
    {
      VariablePtr x;
      LinearConstraintPtr cstr;
      Eigen::Block<Eigen::MatrixXd> block;
    };

    Eigen::DenseIndex m_; //row size of A
    Eigen::DenseIndex n_; //col size of A
    Eigen::DenseIndex r_; //rank of A
    bool constant_;       //constness of A
    bool init_;           //used to perform update_() only once if A is constant
    bool simple_;         //true if there is only one variable and one constraint
    Eigen::MatrixXd A_; //aggregated matrix for non-simple case;
    /** All the pairs (x,c) with x in variables_ and c in constraints_ for which
      * c.contains(x), and the block of A in which to copy c.jacobian(x)
      */
    std::vector<FillData> fillData_;

    friend class SubstitutionCalculator;
  };

  inline Eigen::DenseIndex SubstitutionCalculatorImpl::m() const
  {
    return m_;
  }

  inline Eigen::DenseIndex SubstitutionCalculatorImpl::n() const
  {
    return n_;
  }

  inline Eigen::DenseIndex SubstitutionCalculatorImpl::r() const
  {
    return r_;
  }

  inline bool SubstitutionCalculatorImpl::isSimple() const
  {
    return simple_;
  }

} // internal

} // hint

} // tvm