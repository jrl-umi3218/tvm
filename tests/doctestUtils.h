/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

namespace doctest
{
  /** A utility class to do matrix comparison.
    *
    * Use as in FAST_CHECK_EQ(A, doctest::MApprox(C).precision(1e-6));
    */
  class MApprox
  {
  public:
    template<typename Derived>
    explicit MApprox(const Eigen::DenseBase<Derived>& m)
      : tmp_(), matrix_(m), precision_(Eigen::NumTraits<double>::dummy_precision()) {}

    template<typename Derived>
    explicit MApprox(Eigen::DenseBase<Derived>&& m)
      : tmp_(m), matrix_(tmp_), precision_(Eigen::NumTraits<double>::dummy_precision()) {}

    template<typename Derived>
    explicit MApprox(const Eigen::QuaternionBase<Derived>& m)
      : tmp_(), matrix_(m.coeffs()), precision_(Eigen::NumTraits<double>::dummy_precision()) {}

    template<typename Derived>
    explicit MApprox(Eigen::QuaternionBase<Derived>&& m)
      : tmp_(m.coeffs()), matrix_(tmp_), precision_(Eigen::NumTraits<double>::dummy_precision()) {}

    MApprox& precision(double p)
    {
      precision_ = p;
      return *this;
    }

    DOCTEST_INTERFACE friend bool operator==(Eigen::Ref<const Eigen::MatrixXd> lhs, MApprox const& rhs)
    {
      return lhs.isApprox(rhs.matrix_, rhs.precision_);
    }

    template<typename Derived>
    DOCTEST_INTERFACE friend bool operator==(const Eigen::QuaternionBase<Derived>& lhs, MApprox const& rhs)
    {
      return lhs.coeffs().isApprox(rhs.matrix_, rhs.precision_);
    }

    String toString() const
    {
      return String("\nMApprox(\n") + doctest::toString(matrix_) + "\n)\n";
    }

  private:
    Eigen::MatrixXd tmp_;
    Eigen::Ref<const Eigen::MatrixXd> matrix_;
    double precision_;
  };

  template <>
  inline String toString<MApprox>(const DOCTEST_REF_WRAP(MApprox) value)
  {
    return value.toString();
  }
}