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

  private:
    template<typename Repr>
    static auto logImpl(const Repr& X)
    {
      return X;
    }

    template<typename Tan>
    static auto expImpl(const Tan& t)
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

    friend Base;
  };
}