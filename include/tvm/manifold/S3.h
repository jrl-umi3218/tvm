/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/RotationBase.h>
#include <tvm/manifold/internal/details.h>

namespace tvm::manifold
{
  class S3 : public internal::RotationBase, public internal::LieGroup<S3>
  {
  public:
    using Base = internal::LieGroup<S3>;
    using typename Base::repr_t;
    using typename Base::tan_t;

  private:
    template<typename Repr>
    static tan_t logImpl(const Repr& X)
    {
      const repr_t& q_ = X;
      auto v = q_.vec();
      const auto n2 = v.squaredNorm();
      assert(std::abs(n2 + q_.w() * q_.w() - 1) < 100 * std::numeric_limits<repr_t::Scalar>::epsilon() && "Quaternion must be normalized");

      if (n2 < 1e-28) // sqrt(n2) < 1e-14.
        return (repr_t::Scalar(2) / q_.w()) * v;
      else
      {
        const auto n = std::sqrt(n2);
        return std::atan2(2 * n * q_.w(), q_.w() * q_.w() - n2) / n * q_.vec();
      }
    }

    template<typename Tan>
    static repr_t expImpl(const Tan& t)
    {
      typename Tan::PlainObject u = t;
      const auto n2 = u.squaredNorm();
      repr_t::Scalar c, s;
      if (n2 < tvm::internal::sqrt(std::numeric_limits<repr_t::Scalar>::epsilon()))
      { 
        c = 1 - n2 / 8;
        s = internal::sincsqrt(n2 / 4) / 2;
      }
      else
      {
        const auto n = std::sqrt(n2);
        c = std::cos(n / 2);
        s = std::sin(n / 2) / n;
      }
      return { c, s * u.x(), s * u.y(), s * u.z() };
    }

    template<typename Repr>
    static auto inverseImpl(const Repr& X)
    {
      return X.inverse();
    }

    friend Base;
  };
}