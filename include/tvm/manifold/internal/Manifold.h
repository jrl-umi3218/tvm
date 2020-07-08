/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <type_traits>

#include <Eigen/Core>

#include <tvm/internal/meta.h>

namespace tvm::manifold::internal
{
  template<typename Derived> struct traits {};

  template<typename T> using has_dynamicDim_t = decltype(T::dynamicDim(std::declval<typename T::repr_t>));

  template<typename Derived>
  class Manifold
  {
  public:
    static constexpr int dim = traits<Derived>::dim;

    struct Tangent
    {
      using type = Eigen::Matrix<double, dim, 1>;
    };

    using repr_t = typename traits<Derived>::repr_t;
    using tan_t = typename Tangent::type;

    template<typename Repr>
    static constexpr bool alsoAcceptAsRepr() { return false; }

    template<typename Tan>
    static constexpr bool alsoAcceptAsTan() { return false; }

    static constexpr bool hasStaticDim() { return dim >= 0; }

    template<typename Repr>
    static int dynamicDim(const Repr& X)
    {
      if constexpr (tvm::internal::is_detected<has_dynamicDim_t, traits<Derived>>::value)
        return traits<Derived>::dynamicDim(X);
      else
      {
        static_assert(hasStaticDim(), "The dimension is not known at compile time, "
          "and traits::Derived does not offer a way to know it at runtime (no dynamicDyn function).");
        return dim;
      }

    }

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