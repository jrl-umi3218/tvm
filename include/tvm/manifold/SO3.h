/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/RotationBase.h>

namespace tvm::manifold
{
  /** Group of rotation in 3d represented by orthogonal matrices.*/
  class SO3: public internal::RotationBase, public internal::LieGroup<SO3>
  {
  public:
    using Base = internal::LieGroup<SO3>;
    using typename Base::repr_t;
    using typename Base::tan_t;

  private:
    template<typename Repr>
    static tan_t logImpl(const Repr& X)
    {
      const typename Repr::PlainObject& R_ = X;
      repr_t::Scalar cosTheta = 0.5 * (R_.trace() - 1);
      repr_t::Scalar theta = std::acos(std::min(std::max(cosTheta, -1.), 1.));

      return 0.5 * internal::sinc_inv(theta) * tan_t(R_(2, 1) - R_(1, 2), R_(0, 2) - R_(2, 0), R_(1, 0) - R_(0, 1));
    }

    template<typename Tan>
    static repr_t expImpl(const Tan& t)
    {
      typename Tan::PlainObject u = t;
      const auto n = u.norm();
      if (n < tvm::internal::sqrt(std::numeric_limits<repr_t::Scalar>::epsilon()))
        return repr_t::Identity() + hat(u);

      u /= n;
      repr_t::Scalar c = std::cos(n);
      repr_t::Scalar s = std::sin(n);
      repr_t X = (1 - c) * u * u.transpose(); // (1-c)uu^t
      X += hat(s * u);                        // + s*hat(u)
      X.diagonal().array() += c;              // + c*I 
      return X;
    }

    template<typename Repr>
    static auto inverseImpl(const Repr& X)
    {
      return X.transpose();
    }

    friend Base;
  };
}