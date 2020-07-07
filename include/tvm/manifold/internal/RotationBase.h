/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/Adjoint.h>
#include <tvm/manifold/internal/LieGroup.h>

#include <Eigen/Geometry>

namespace tvm::manifold 
{
  class SO3;
  class S3;

  namespace internal
  {
    template<>
    struct traits<SO3>
    {
      static constexpr int dim = 3;
      using repr_t = Eigen::Matrix3d;
    };

    template<>
    struct AdjointOperations<SO3>
    {
      using matrix_t = Eigen::Matrix3d;
      static const auto& toMatrix(const typename traits<SO3>::repr_t& X) { return X; }
    };
  }

  namespace internal
  {
    template<>
    struct traits<S3>
    {
      static constexpr int dim = 3;
      using repr_t = Eigen::Quaternion<double>;
    };

    template<>
    struct AdjointOperations<S3>
    {
      using matrix_t = Eigen::Matrix3d;
      static auto toMatrix(const typename traits<S3>::repr_t& X) { return X.toRotationMatrix(); }
    };
  }
}


namespace tvm::manifold::internal
{
  class RotationBase
  {
  public:
    using tan_t = typename internal::LieGroup<SO3>::tan_t;

    template<typename Tan>
    static Eigen::Matrix3d hat(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t>);
      const typename Tan::PlainObject& t_ = t;
      Eigen::Matrix3d mat;
      mat << 0., -t_.z(), t_.y(),
        t_.z(), 0., -t_.x(),
        -t_.y(), t_.x(), 0.;
      return mat;
    }

    template<typename Mat>
    static tan_t vee(const Mat& M)
    {
      static_assert(std::is_convertible_v<Mat, Eigen::Matrix3d>);
      return { M(2,1), M(0,2), M(1,0) };
    }

  protected:
    template<typename ReprX, typename ReprY>
    static auto composeImpl(const ReprX& X, const ReprY& Y)
    {
      return X * Y;
    }
  };
}