/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/Adjoint.h>
#include <tvm/manifold/internal/details.h>
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
      mat <<    0., -t_.z(),  t_.y(),
            t_.z(),   0.   , -t_.x(),
           -t_.y(),  t_.x(),   0.;
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

    template<typename Tan, typename Which>
    static auto jacobianImpl(const Tan& t, Which)
    {
      const typename Tan::PlainObject& u = t;
      auto nu = u.norm();

      Eigen::Matrix3d C2;
      auto [f1,f2] = internal::SO3JacF1F2(nu);
      auto xx = f2 * u.x() * u.x();
      auto xy = f2 * u.x() * u.y();
      auto xz = f2 * u.x() * u.z();
      auto yy = f2 * u.y() * u.y();
      auto yz = f2 * u.y() * u.z();
      auto zz = f2 * u.z() * u.z();
      C2 << -yy - zz, xy, xz,
             xy, -xx - zz, yz,
             xz, yz, -xx - yy;

      Eigen::Matrix3d C = hat(Eigen::Vector3d(f1*u));
      auto I = Eigen::Matrix3d::Identity();
      
      if constexpr (std::is_same_v<Which, right_t>)
        return Eigen::Matrix3d(I - C + C2);
      else if constexpr (std::is_same_v<Which, left_t>)
        return Eigen::Matrix3d(I + C + C2);
      else
      {
        static_assert(std::is_same_v<Which, both_t>);
        return std::make_pair(Eigen::Matrix3d(I + C + C2), Eigen::Matrix3d(I - C + C2));
      }
    }

    template<typename Tan, typename Which>
    static auto invJacobianImpl(const Tan& t, Which)
    {
      const typename Tan::PlainObject& u = t;
      auto nu = u.norm();

      Eigen::Matrix3d C2;
      auto f2 = internal::SO3JacInvF2(nu);
      auto xx = f2 * u.x() * u.x();
      auto xy = f2 * u.x() * u.y();
      auto xz = f2 * u.x() * u.z();
      auto yy = f2 * u.y() * u.y();
      auto yz = f2 * u.y() * u.z();
      auto zz = f2 * u.z() * u.z();
      C2 << -yy - zz, xy, xz,
             xy, -xx - zz, yz,
             xz, yz, -xx - yy;

      Eigen::Matrix3d C = hat(Eigen::Vector3d(u / 2));
      auto I = Eigen::Matrix3d::Identity();

      if constexpr (std::is_same_v<Which, right_t>)
        return Eigen::Matrix3d(I + C + C2);
      else if constexpr (std::is_same_v<Which, left_t>)
        return Eigen::Matrix3d(I - C + C2);
      else
      {
        static_assert(std::is_same_v<Which, both_t>);
        return std::make_pair(Eigen::Matrix3d(I + C + C2), Eigen::Matrix3d(I - C + C2));
      }
    }
  };
}