/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include <tvm/hint/internal/GenericCalculator.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

#include <sstream>

using namespace Eigen;

namespace tvm
{

namespace hint
{
    
namespace internal
{
  std::unique_ptr<abstract::SubstitutionCalculatorImpl> GenericCalculator::impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const
  {
    return std::unique_ptr<abstract::SubstitutionCalculatorImpl>(new GenericCalculator::Impl(cstr, x, rank));
  }

  GenericCalculator::Impl::Impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank)
    : SubstitutionCalculatorImpl(cstr, x, rank)
    , qr_(m(),n())
    , invR1R2_(r(), n()-r())
    , tmp_(m(), 2*n())
  {
  }

  void GenericCalculator::Impl::update_()
  {
    if (isSimple())
    {
      qr_.compute(constraints_.front()->jacobian(*variables_[0]));
    }
    else
    {
      fillA();
      qr_.compute(A());
    }

    if (qr_.rank() != r())
    {
      std::stringstream ss;
      const auto& vars = variables_.variables();
      ss << "During the substitution of the ";
      if (variables_.variables().size() == 1)
      {
        ss << "variable " << vars.front()->name();
      }
      else
      {
        ss << "variables (";
        for (size_t i = 0; i < vars.size() - 1; ++i)
        {
          ss << vars[i]->name() << ", ";
        }
        ss << vars.back()->name() << ")";
      }
      ss << ": the rank of the matrix (" << qr_.rank();
      ss << ") is not the one that was specified (" << r() << ").";
      throw std::runtime_error(ss.str());
    }
      
    //Set shortcuts
    const auto& R1 = qr_.matrixR().topLeftCorner(r(), r()).template triangularView<Eigen::Upper>();
    auto& R2 = qr_.matrixR().topRightCorner(r(), n() - r());
    const auto& P = qr_.colsPermutation().indices();

    //Compute inv(R1) * R2
    invR1R2_ = R1.solve(R2);

    // N = P_2 - P_1 * inv(R1) * R2
    for (Eigen::DenseIndex i = 0; i < r(); ++i)
    {
      N_.row(P.coeff(i)) = -invR1R2_.row(i);
    }
    for (auto i = r(); i < n(); ++i)
    {
      N_.row(P.coeff(i)).setZero();
      N_(P.coeff(i), i - r()) = 1;
    }
  }

  void GenericCalculator::Impl::premultiplyByASharpAndSTranspose_(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const
  {
    //For M = in, and A = | Q1   Q2 | | R1  R2 | | P1^T | 
    //                                |  0   0 | | P2^T |
    // we compute | P1 R1^-1   0 | | Q1^T | M
    //            |    0       I | | Q2^T |     
    tmp_.resize(m(), in.cols());
    auto T = tmp_.get();
    
    // T = Q^T M
    T = qr_.matrixQ().setLength(qr_.nonzeroPivots()).transpose() * in;

    // T(1:r,:) = R1^-1 T(1:r,:)
    const auto& R1 = qr_.matrixR().topLeftCorner(r(), r()).template triangularView<Eigen::Upper>();
    R1.solveInPlace(T.topRows(r()));

    // outA = - P1 * T(1:r,:)
    if (minus)
    {
      for (DenseIndex i = 0; i < r(); ++i)
      {
        outA.row(qr_.colsPermutation().indices().coeff(i)) = -T.row(i);
      }
    }
    else
    {
      for (DenseIndex i = 0; i < r(); ++i)
      {
        outA.row(qr_.colsPermutation().indices().coeff(i)) = T.row(i);
      }
    }
    
    for (DenseIndex i = r(); i < n(); ++i)
    {
      outA.row(qr_.colsPermutation().indices().coeff(i)).setZero();
    }

    // outS
    outS = T.bottomRows(m() - r());
  }

}

}

}
