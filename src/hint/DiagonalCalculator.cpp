#include <tvm/hint/internal/DiagonalCalculator.h>
#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

#include <algorithm>

using namespace Eigen;

namespace
{
  /** max value of two sorted vectors, where the second one can be empty*/
  DenseIndex optionalMax(const std::vector<DenseIndex>& v1, const std::vector<DenseIndex>& v2)
  {
    assert(v1.size() > 0);
    if (v2.empty())
    {
      return v1.back();
    }
    else
    {
      return std::max(v1.back(), v2.back());
    }
  }
}

namespace tvm
{

namespace hint
{

namespace internal
{
  DiagonalCalculator::DiagonalCalculator(DenseIndex first, DenseIndex size)
    : first_(first), size_(size)
  {
    if (first < 0)
    {
      throw std::runtime_error("first must be non-negative.");
    }
    if (size < -1)
    {
      throw std::runtime_error("Invalid size.");
    }
  }

  DiagonalCalculator::DiagonalCalculator(const std::vector<DenseIndex>& nnzRows, const std::vector<DenseIndex>& zeroRows)
    : first_(-1), size_(-1), nnz_(nnzRows), zeros_(zeroRows)
  {
    if (nnz_.empty())
    {
      throw std::runtime_error("There should be at least one non-zero row.");
    }

    std::sort(nnz_.begin(), nnz_.end());
    std::sort(zeros_.begin(), zeros_.end());

    std::vector<bool> appear(optionalMax(nnz_, zeros_) + 1, false);

    for (auto i : nnz_)
    {
      if (i < 0)
      {
        throw std::runtime_error("Indices must be non-negative.");
      }
      if (appear[static_cast<size_t>(i)])
      {
        throw std::runtime_error("Each index must be unique.");
      }
      else
      {
        appear[i] = true;
      }
    }

    for (auto i : zeros_)
    {
      if (i < 0)
      {
        throw std::runtime_error("Indices must be non-negative.");
      }
      if (appear[static_cast<size_t>(i)])
      {
        throw std::runtime_error("Each index of nnzRows and zeroRows must be unique.");
      }
      else
      {
        appear[i] = true;
      }
    }
  }

  std::unique_ptr<abstract::SubstitutionCalculatorImpl> DiagonalCalculator::impl_(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank) const
  {
    if (first_ >= 0)
    {
      return std::unique_ptr<abstract::SubstitutionCalculatorImpl>(new Impl(cstr, x, rank, first_, size_));
    }
    else
    {
      return std::unique_ptr<abstract::SubstitutionCalculatorImpl>(new Impl(cstr, x, rank, nnz_, zeros_));
    }
  }

  DiagonalCalculator::Impl::Impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank, DenseIndex first, DenseIndex size)
    : SubstitutionCalculatorImpl(cstr, x, rank)
    , first_(first)
    , size_(size)
  {
    if (!isSimple())
    {
      throw std::runtime_error("This calculator is only for the substitution of a single variable through a single constraint.");
    }
    if (size == -1)
    {
      size_ = x.front()->size() - first;
    }
    if (size_ != rank)
    {
      throw std::runtime_error("The number of non-zero rows is not consistent with the rank.");
    }

    DenseIndex i = 0;
    for (; i < first; ++i)
    {
      cnnz_.push_back(i);
    }
    for (DenseIndex j = 0; j < size_; ++i, ++j)
    {
      nnz_.push_back(i);
    }
    for (; i < n(); ++i)
    {
      cnnz_.push_back(i);
    }


    build();
  }

  DiagonalCalculator::Impl::Impl(const std::vector<LinearConstraintPtr>& cstr, const std::vector<VariablePtr>& x, int rank, const std::vector<DenseIndex>& nnzRows, const std::vector<DenseIndex>& zeroRows)
    : SubstitutionCalculatorImpl(cstr, x, rank)
    , first_(-1)
    , size_(0)
    , nnz_(nnzRows)
    , cnnz_()
    , zeros_(zeroRows)
  {
    if (!isSimple())
    {
      throw std::runtime_error("This calculator is only for the substitution of a single variable through a single constraint.");
    }
    if (static_cast<int>(optionalMax(nnzRows, zeroRows)) >= n())
    {
      throw std::runtime_error("Some indices are too large for the substitution considered");
    }
    if (static_cast<int>(nnzRows.size()) != rank)
    {
      throw std::runtime_error("The number of non-zero rows is not consistent with the rank.");
    }

    // build cnnz_
    for (DenseIndex i = 0; i < nnz_.front(); ++i)
    {
      cnnz_.push_back(i);
    }
    for (size_t j = 0; j < nnz_.size()-1; ++j)
    {
      for (auto i = nnz_[j] + 1; i < nnz_[j + 1]; ++i)
      {
        cnnz_.push_back(i);
      }
    }
    for (auto i = nnz_.back() + 1; i < n(); ++i)
    {
      cnnz_.push_back(i);
    }

    build();
  }

  void DiagonalCalculator::Impl::update_()
  {
    const auto& A = constraints_[0]->jacobian(*variables_[0]);
    for (size_t i = 0; i < nnz_.size(); ++i)
    {
      inverse_[static_cast<DenseIndex>(i)] = 1. / A(innz_[i], nnz_[i]);
    }
  }

  void DiagonalCalculator::Impl::premultiplyByASharpAndSTranspose_(MatrixRef outA, MatrixRef outS, const MatrixConstRef& in, bool minus) const
  {
    //FIXME: optimized version for identity/minus identity

    // multiplying by A^#
    if (minus)
    {
      for (size_t i = 0; i < nnz_.size(); ++i)
      {
        outA.row(nnz_[i]) = -inverse_[static_cast<DenseIndex>(i)] * in.row(innz_[i]);
      }
    }
    else
    {
      for (size_t i = 0; i < nnz_.size(); ++i)
      {
        outA.row(nnz_[i]) = inverse_[static_cast<DenseIndex>(i)] * in.row(innz_[i]);
      }
    }
    for (auto i : cnnz_)
    {
      outA.row(i).setZero();
    }

    //multiplying by S^T
    for (size_t i = 0; i < zeros_.size(); ++i)
    {
      outS.row(static_cast<DenseIndex>(i)) = in.row(zeros_[i]);
    }
  }

  void DiagonalCalculator::Impl::postMultiplyByN_(MatrixRef out, const MatrixConstRef& in, bool add) const
  {
    if (add)
    {
      for (size_t i = 0; i < cnnz_.size(); ++i)
      {
        out.col(static_cast<DenseIndex>(i)) += in.col(cnnz_[i]);
      }
    }
    else
    {
      for (size_t i = 0; i < cnnz_.size(); ++i)
      {
        out.col(static_cast<DenseIndex>(i)) = in.col(cnnz_[i]);
      }
    }
  }

  void DiagonalCalculator::Impl::build()
  {
    N_.setZero();
    for (size_t i = 0; i < cnnz_.size(); ++i)
    {
      N_(cnnz_[i], static_cast<DenseIndex>(i)) = 1;
    }

    std::vector<bool> appearnnz(static_cast<size_t>(n()), false);
    std::vector<bool> appearZero(static_cast<size_t>(n()), false);
    for (auto i : nnz_)
    {
      appearnnz[i] = true;
    }
    for (auto i : zeros_)
    {
      appearZero[i] = true;
    }
    DenseIndex k = 0;
    for (size_t i = 0; i < static_cast<size_t>(n()); ++i)
    {
      if (appearnnz[i])
      {
        innz_.push_back(k);
        ++k;
      }
      else if (appearZero[i])
      {
        ++k;
      }
    }

    inverse_.resize(static_cast<DenseIndex>(nnz_.size()));
  }
}
}
}