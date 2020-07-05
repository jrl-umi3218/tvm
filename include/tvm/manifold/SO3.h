/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/LieGroup.h>
#include <tvm/manifold/internal/details.h>

namespace tvm::manifold
{
  class SO3;

  namespace internal
  {
    template<>
    struct traits<SO3>
    {
      static constexpr int dim = 3;
      using repr_t = Eigen::Matrix3d;
    };
  }


  class SO3: public internal::LieGroup<SO3>
  {
  public:
    using Base = internal::LieGroup<SO3>;
    using typename Base::repr_t;
    using typename Base::tan_t;

    template<typename Tan>
    static Eigen::Matrix3d hat(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t>);
      const typename Tan::PlainObject& t_ = t;
      Eigen::Matrix3d mat;
      mat << 0., -t_.z(),  t_.y(), 
         t_.z(),    0.  , -t_.x(), 
        -t_.y(),  t_.x(),    0.;
      return mat;
    }

    template<typename Mat>
    static tan_t vee(const Mat& M)
    {
      static_assert(std::is_convertible_v<Mat, Eigen::Matrix3d>);
      return {M(2,1), M(0,2), M(1,0)};
    }

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
      u /= n;
      repr_t::Scalar c = std::cos(n);
      repr_t::Scalar s = std::sin(n);
      repr_t X = (1 - c) * u * u.transpose(); // (1-c)uu^t
      X += hat(s * u);                        // + s*hat(u)
      X.diagonal().array() += c;              // + c*I 
      return X;
    }

    template<typename ReprX, typename ReprY>
    static auto composeImpl(const ReprX& X, const ReprY& Y)
    {
      return X * Y;
    }

    template<typename Repr>
    static auto inverseImpl(const Repr& X)
    {
      return X.transpose();
    }

    friend Base;
  };
}