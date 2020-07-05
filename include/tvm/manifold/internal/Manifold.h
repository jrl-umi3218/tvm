/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <Eigen/Core>

#include <type_traits>

namespace tvm::manifold::internal
{
  template<typename Derived> struct traits {};

  template<typename Derived>
  class Manifold
  {
  public:
    static constexpr int dim = traits<Derived>::dim;
    using repr_t = typename traits<Derived>::repr_t;
    using tan_t = Eigen::Matrix<double, dim, 1>;


    template<typename Repr>
    static constexpr bool alsoAcceptAsRepr() { return false; }

    template<typename Tan>
    static constexpr bool alsoAcceptAsTan() { return false; }

    static constexpr bool FiniteDim() { return dim > 0; }

    template<typename Repr>
    static auto log(const Repr& X)
    {
      using ret_t = decltype(Derived::log_(X));
      static_assert(std::is_convertible_v<Repr, repr_t> || Derived::template alsoAcceptAsRepr<Repr>());
      static_assert(std::is_convertible_v<ret_t, tan_t> || Derived::template alsoAcceptAsTan<ret_t>());
      return Derived::log_(X);
    }

    template<typename Tan>
    static auto exp(const Tan& t)
    {
      using ret_t = decltype(Derived::exp_(t));
      static_assert(std::is_convertible_v<Tan, tan_t> || Derived::template alsoAcceptAsTan<Tan>());
      static_assert(std::is_convertible_v<ret_t, repr_t> || Derived::template alsoAcceptAsRepr<ret_t>());
      return Derived::exp_(t);
    }

  };
}