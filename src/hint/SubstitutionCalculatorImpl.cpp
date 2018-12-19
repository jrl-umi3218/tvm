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

#include <tvm/hint/abstract/SubstitutionCalculatorImpl.h>

#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

namespace
{
  // return true if the buffers for M1 and M2 are disjoint
  bool noAliasing(const tvm::MatrixConstRef& M1, const tvm::MatrixConstRef& M2)
  {
    return M1.data() >= M2.data() + M2.size() || M2.data() >= M1.data() + M1.size();
  }
}

namespace tvm
{

namespace hint
{

namespace abstract
{

void SubstitutionCalculatorImpl::update()
{
  if (!constant_ || init_)
  {
    update_();
    assert(N_.rows() == n_);
    if (N_.cols() != n_ - r_)
    {
      throw std::runtime_error("N_ does not have the correct size. Did you specify a correct rank for the substitution, or did the rank change (which is not allowed)?");
    }
    init_ = false;
  }
}

void SubstitutionCalculatorImpl::premultiplyByASharpAndSTranspose(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const
{
  assert(noAliasing(outA, in));
  assert(noAliasing(outS, in));
  assert(in.rows() == m_);
  assert(outA.rows() == n_);
  assert(outA.cols() == in.cols());
  assert(outS.rows() == m_ - r_);
  assert(outS.cols() == in.cols());

  premultiplyByASharpAndSTranspose_(outA, outS, in, minus);

  assert(outA.rows() == n_);
  assert(outA.cols() == in.cols());
  assert(outS.rows() == m_ - r_);
  assert(outS.cols() == in.cols());
}

void SubstitutionCalculatorImpl::postMultiplyByN(MatrixRef out, const MatrixConstRef& in, bool add) const
{
  assert(noAliasing(out, in));
  assert(out.cols() == n_ - r_);
  assert(in.cols() == n_);
  assert(out.rows() == in.rows());

  postMultiplyByN_(out, in, add);

  assert(out.cols() == n_ - r_);
  assert(out.rows() == in.rows());
}

void SubstitutionCalculatorImpl::postMultiplyByN(MatrixRef out, const MatrixConstRef & in, Range r, bool add) const
{
  assert(noAliasing(out, in));
  assert(out.cols() == n_ - r_);
  assert(in.cols() == r.dim);
  assert(out.rows() == in.rows());

  if (r.start == 0 && r.dim == n_)
  {
    postMultiplyByN_(out, in, add);
  }
  else
  {
    postMultiplyByN_(out, in, r, add);
  }

  assert(out.cols() == n_ - r_);
  assert(out.rows() == in.rows());
}

void SubstitutionCalculatorImpl::postMultiplyByN_(MatrixRef out, const MatrixConstRef & in, bool add) const
{
  if (add)
  {
    out.noalias() += in * N();
  }
  else
  {
    out.noalias() = in * N();
  }
}

void SubstitutionCalculatorImpl::postMultiplyByN_(MatrixRef out, const MatrixConstRef & in, Range r, bool add) const
{
  if (add)
  {
    out.noalias() += in * N().middleRows(r.start,r.dim);
  }
  else
  {
    out.noalias() = in * N().middleRows(r.start, r.dim);
  }
}

void SubstitutionCalculatorImpl::fillA()
{
  for (auto& f : fillData_)
  {
    f.block = f.cstr->jacobian(*f.x);
  }
}

const Eigen::MatrixXd& SubstitutionCalculatorImpl::N() const
{
  return N_;
}

SubstitutionCalculatorImpl::SubstitutionCalculatorImpl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank)
  : constraints_(cstr)
  , variables_(x)
  , m_(0)
  , n_(0)
  , r_(rank)
  , constant_(false)
  , init_(true)
  , simple_(cstr.size()==1 && x.size()==1)
{
  assert(rank >= 0);

  for (const auto& c : cstr)
  {
    m_ += c->size();
  }

  for (const auto& v : x)
  {
    n_ += v->size();
  }

  assert(m_ <= n_);
  assert(r_ <= m_);

  if (!isSimple())
  {
    A_.resize(m_, n_);
    A_.setZero();

    //build fillData
    int m = 0;
    for (auto c : cstr)
    {
      int mi = c->size();
      for (auto v : variables_.variables())
      {
        if (c->variables().contains(*v))
        {
          auto cols = v->getMappingIn(variables_);
          fillData_.push_back({ v, c, A_.block(m,cols.start,mi,cols.dim) });
        }
      }
      m += mi;
    }
  }

  //check constness
  constant_ = true;
  for (const auto& c : constraints_)
  {
    for (const auto& v : variables_.variables())
    {
      if (c->variables().contains(*v))
      {
        constant_ &= c->jacobian(*v).properties().isConstant();
      }
    }
  }

  N_.resize(n_, n_ - r_);
}

void SubstitutionCalculatorImpl::constant(bool c)
{
  constant_ = c;
}

bool SubstitutionCalculatorImpl::constant() const
{
  return constant_;
}

const Eigen::MatrixXd & SubstitutionCalculatorImpl::A() const
{
  if (isSimple())
  {
    return constraints_[0]->jacobian(*variables_[0]);
  }
  else
  {
    return A_;
  }
}

}

}

}
