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

  class TVM_DLLAPI SubstitutionCalculatorImpl
  {
  public:
    void update();
    void premultiplyByASharpAndSTranspose(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const;
    void postMultiplyByN(MatrixRef out, const MatrixConstRef& in, bool add) const;
    const Eigen::MatrixXd& N() const;

    Eigen::DenseIndex m() const;
    Eigen::DenseIndex n() const;
    Eigen::DenseIndex r() const;

  protected:
    SubstitutionCalculatorImpl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank);
    
    void constant(bool c);
    bool constant() const;

    const Eigen::MatrixXd& A() const;

    /** Return true if there is only one variable and one constraint*/
    bool isSimple() const;
    
    virtual void update_() = 0;
    virtual void premultiplyByASharpAndSTranspose_(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const = 0;
    virtual void postMultiplyByN_(MatrixRef out, const MatrixConstRef& in, bool add) const = 0;

    void fillA();

    Eigen::MatrixXd N_;
    std::vector<LinearConstraintPtr> constraints_;
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
    bool constant_;
    bool init_;
    bool simple_;
    Eigen::MatrixXd A_; //aggregated matrix for non-simple case;
    /** All the pairs (x,c) with x in variables_ and c in constraints_ for which
      * c.contains(x), and the block of A in which to copy c.jacobian(x)*/
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

  inline const Eigen::MatrixXd & SubstitutionCalculatorImpl::A() const
  {
    return A_;
  }

  inline bool SubstitutionCalculatorImpl::isSimple() const
  {
    return simple_;
  }

} // internal

} // hint

} // tvm