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

  /** Base class for all manifolds, including non Lie-group manifolds like S2.
    *
    * Manifold classes are meant to provide only static members and functions.
    *
    * \internal For a first version, I chose to take only double as scalars, to
    * avoid too many complications at once. However, in many places I tried to
    * use a generic approach to allow for east generalization.
    */
  template<typename Derived>
  class Manifold
  {
  public:
    /** Dimension of the manifold. Negative numbers indicate a dimension not
      * known at runtime. The general idea is to use Eigen::Dynamic for that.*/
    static constexpr int dim = traits<Derived>::dim;

    /** Informations about the tangent space of a given manifolds
      *
      * \internal A more advanced implementation of the lib could consider making
      * Tangent a full-fledge class with LieAlgebra deriving of it.
      */
    struct Tangent
    {
      // Type to represent an element of the tangent space viewed as R^n.
      using type = Eigen::Matrix<double, dim, 1>;
    };

    /** Type to represent an element of the manifold*/
    using repr_t = typename traits<Derived>::repr_t;
    /** Type to reprensent an element of the tangent space*/
    using tan_t = typename Tangent::type;

    /** By default, methods with elements of the manifold as arguments only accept
      * type that are convertible to repr_t. This method, if overloaded, can be
      * used to accept additional types. We use it in particular to accept Identity. 
      */
    template<typename Repr>
    static constexpr bool alsoAcceptAsRepr() { return false; }

    /** Same as alsoAcceptAsRepr, but for arguments in tangent space.*/
    template<typename Tan>
    static constexpr bool alsoAcceptAsTan() { return false; }

    /** Returns true if the dimension of the space is known at compile time.*/
    static constexpr bool hasStaticDim() { return dim >= 0; }

    /** Given an element of the manifold, returns the size of the manifold.
      * This function is useful for manifolds whose size is not known at
      * compile time
      */
    template<typename Repr>
    static int dynamicDim(const Repr& X)
    {
      static_assert(std::is_convertible_v<Repr, repr_t> || Derived::template alsoAcceptAsRepr<Repr>());
      if constexpr (tvm::internal::is_detected<has_dynamicDim_t, traits<Derived>>::value)
        return traits<Derived>::dynamicDim(X);
      else
      {
        static_assert(hasStaticDim(), "The dimension is not known at compile time, "
          "and traits::Derived does not offer a way to know it at runtime (no dynamicDyn function).");
        return dim;
      }

    }

    /** Log function, from manifold to tangent space.
      *
      * The operation in itself is defined by Derived::log_
      */
    template<typename Repr>
    static auto log(const Repr& X)
    {
      using ret_t = decltype(Derived::log_(X));
      // Should we enforce size at compile time here?
      static_assert(std::is_convertible_v<Repr, repr_t> || Derived::template alsoAcceptAsRepr<Repr>());
      static_assert(std::is_convertible_v<ret_t, tan_t> || Derived::template alsoAcceptAsTan<ret_t>());
      return Derived::log_(X);
    }

    /** Exp function, from tangent space to manifold.
      *
      * The operation in itself is defined by Derived::exp_
      */
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