/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/Adjoint.h>
#include <tvm/manifold/internal/LieGroup.h>

namespace tvm::manifold
{
  template<int> class Real;

  namespace internal
  {
    template<int N>
    struct traits<Real<N>>
    {
      static constexpr int dim = N;
      using repr_t = Eigen::Matrix<double, N, 1>;
      template<typename Repr>
      static int dynamicDim(const Repr& X)
      {
        if constexpr (dim >= 0)
          return dim;
        else
          return static_cast<int>(X.size());
      }
    };

    template<int N>
    struct AdjointOperations<Real<N>>
    {
      using matrix_t = Eigen::Matrix<double, N, N>;
      static auto toMatrix(const typename traits<Real<N>>::repr_t& X) { return matrix_t::Identity(X.size(), X.size()); }
    };
  }

  template<int N> 
  class Real: public internal::LieGroup<Real<N>>
  {
  public:
    using Base = internal::LieGroup<Real<N>>;
    using typename Base::repr_t;
    using typename Base::tan_t;
    using typename Base::jacobian_t;

  private:
    template<typename Tan>
    static const Tan& hatImpl(const Tan& t)
    {
      return t;
    }

    template<typename Mat>
    static const Mat& veeImpl(const Mat& M)
    {
      return M;
    }

    template<typename Repr>
    static const auto& logImpl(const Repr& X)
    {
      return X;
    }

    template<typename Tan>
    static const auto& expImpl(const Tan& t)
    {
      return t;
    }

    template<typename ReprX, typename ReprY>
    static auto composeImpl(const ReprX& X, const ReprY& Y)
    {
      return X + Y;
    }

    template<typename Repr>
    static auto inverseImpl(const Repr& X)
    {
      return -X;
    }

    template<typename Tan, typename Which>
    static auto jacobianImpl(const Tan& t, Which)
    {
      if constexpr (std::is_same_v<Which, internal::right_t> || std::is_same_v<Which, internal::left_t>)
        return jacobian_t::Identity(t.size(), t.size());
      else
      {
        static_assert(std::is_same_v<Which, internal::both_t>);
        return std::make_pair(jacobian_t::Identity(t.size(), t.size()), jacobian_t::Identity(t.size(), t.size()));
      }
    }

    template<typename Tan, typename Which>
    static auto invJacobianImpl(const Tan& t, Which)
    {
      if constexpr (std::is_same_v<Which, internal::right_t> || std::is_same_v<Which, internal::left_t>)
        return jacobian_t::Identity(t.size(), t.size());
      else
      {
        static_assert(std::is_same_v<Which, internal::both_t>);
        return std::make_pair(jacobian_t::Identity(t.size(), t.size()), jacobian_t::Identity(t.size(), t.size()));
      }
    }

    friend Base;
  };
}