/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/Adjoint.h>
#include <tvm/manifold/internal/Manifold.h>

namespace tvm::manifold::internal
{
  // Forward declaration
  template<typename Derived> class LieGroup;

  /** Specialization of traits for LieGroup*/
  template<typename Derived>
  struct traits<LieGroup<Derived>>: public traits<Derived>
  {
  };

  /** Dummy structures to specifies "sides" e.g when talking about jacobians.*/
  struct right_t {};
  struct left_t {};
  struct both_t {};

  /** Check if T is equal to right_t. T can only be right_t or left_t*/
  template<typename T>
  static constexpr bool isLocal()
  {
    static_assert(std::is_same_v<T, right_t> || std::is_same_v<T, left_t>, "Invalid type: righ_t or left_t expected.");
    return std::is_same_v<T, internal::right_t>;
  }

  /** This class serves as a base class for all Lie groups.
    * It introduces the composition operation, the inverse and the identity, as
    * well as the notions of ajoint and (manifold) jacobians.
    *
    * \internal In particular, the class handles the cases when arguments of
    * methods are the identity.
    */
  template<typename Derived>
  class LieGroup : public Manifold<LieGroup<Derived>>
  {
  public:
    /** Identity of the group*/
    struct Identity { using LG = Derived;  };
    /** Identity of the algebra*/
    struct AlgIdentity { using LG = Derived; };

    static constexpr auto identity = Identity{};
    static constexpr auto algIdentity = AlgIdentity{};

    using Base = Manifold<LieGroup<Derived>>;
    using typename Base::repr_t;
    using typename Base::tan_t;
    using Base::dim;
    using jacobian_t = Eigen::Matrix<typename tan_t::Scalar, dim, dim>;

  private:
    static constexpr right_t rOnly = {};
    static constexpr left_t lOnly = {};
    static constexpr both_t both = {};

  public:
    /** General compose function X o Y.
      *
      * The operation is carried out by Derived::composeImpl.
      */
    template<typename ReprX, typename ReprY>
    static auto compose(const ReprX& X, const ReprY& Y)
    {
      static_assert(std::is_convertible_v<ReprX, repr_t>);
      static_assert(std::is_convertible_v<ReprY, repr_t>);
      static_assert(std::is_convertible_v<decltype(Derived::composeImpl(X, Y)), repr_t>);
      return Derived::composeImpl(X, Y);
    }

    /** Composition X o Y when Y is the identity.*/
    template<typename Repr>
    static auto compose(const Repr& X, const Identity&)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      return X;
    }

    /** Composition X o Y when X is the identity.*/
    template<typename Repr>
    static auto compose(const Identity&, const Repr& Y)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      return Y;
    }

    /** Composition of two identities.*/
    static auto compose(const Identity&, const Identity&)
    {
      return Identity{};
    }

    /** Inverse of element X, general case.*/
    template<typename Repr>
    static auto inverse(const Repr& X)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      return Derived::inverseImpl(X);
    }

    /** Inverse of the identity.*/
    static auto inverse(const Identity& X)
    {
      return X;
    }

    /** Return the adjoint of X. 
      *
      * If X is of type \p repr_t and not a temporary, the adjoint takes a
      * reference on it.
      *
      * \internal There is definitely room for improvement here, to make this
      * behavior air-tight.
      */
    template<typename Repr>
    static auto adjoint(const Repr& X)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      if constexpr (std::is_same_v<Repr, repr_t>)
        return Adjoint<Derived, ReprRef<Derived>>(X);
      else
        return Adjoint<Derived>(repr_t(X));
    }

    /** Return the adjoint of a temporary element*/
    static auto adjoint(repr_t&& X)
    {
      return Adjoint<Derived>(std::forward<repr_t>(X));
    }

    /** Ajoint of identity.*/
    static auto adjoint(const Identity& X)
    {
      return Adjoint(X);
    }

    /** Compute the right jacobian Jr of the manifold at \p t (i.e. the derivative
      * of exp(t) w.r.t \p t, where small variations of \p t are mapped to small
      * variation of X = exp(t) in local tangent space: exp(t+dt) ~ X exp(Jr(t) dt).
      */
    template<typename Tan>
    static auto rightJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::jacobianImpl(t, rOnly);
    }

    /** Compute the left jacobian Jl of the manifold at \p t (i.e. the derivative
      * of exp(t) w.r.t \p t, where small variations of \p t are mapped to small
      * variation of X = exp(t) in global tangent space: exp(t+dt) ~ exp(Jl(t) dt) X.
      */
    template<typename Tan>
    static auto leftJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::jacobianImpl(t, lOnly);
    }

    /** Compute both right and left jacobians, returned as a pair (right, left).*/
    template<typename Tan>
    static auto jacobians(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::jacobianImpl(t, both);
    }

    /** Compute the inverse of the right jacobian.*/
    template<typename Tan>
    static auto invRightJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::invJacobianImpl(t, rOnly);
    }

    /** Compute the inverse of the left jacobian.*/
    template<typename Tan>
    static auto invLeftJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::invJacobianImpl(t, lOnly);
    }

    /** Compute the inverse of both jacobians, return as a pair (right, left).*/
    template<typename Tan>
    static auto invJacobians(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::invJacobianImpl(t, both);
    }

  private:
    /** We want methods of manifold to accept also identity.*/
    template<typename Repr>
    static constexpr bool alsoAcceptAsRepr() { return std::is_same_v<Repr, Identity>; }

    /** We want methods of manifold to accept also identity.*/
    template<typename Tan>
    static constexpr bool alsoAcceptAsTan() { return std::is_same_v<Tan, AlgIdentity>; }

    template<typename Repr>
    static auto log_(const Repr& X)
    {
        return Derived::logImpl(X);
    }

    static AlgIdentity log_(const Identity&) { return {}; }

    template<typename Tan>
    static auto exp_(const Tan& X)
    {
      return Derived::expImpl(X);
    }

    static Identity exp_(const AlgIdentity&) { return {}; }

    friend Base;
  };
}