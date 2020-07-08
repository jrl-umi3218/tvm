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


  struct right_t {};
  struct left_t {};
  struct both_t {};

  /** This class serves as a base class for all Lie groups.
    * It introduces the composition operation, the inverse
    * and the identity.
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
    template<typename ReprX, typename ReprY>
    static auto compose(const ReprX& X, const ReprY& Y)
    {
      static_assert(std::is_convertible_v<ReprX, repr_t>);
      static_assert(std::is_convertible_v<ReprY, repr_t>);
      static_assert(std::is_convertible_v<decltype(Derived::composeImpl(X, Y)), repr_t>);
      return Derived::composeImpl(X, Y);
    }

    template<typename Repr>
    static auto compose(const Repr& X, const Identity&)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      return X;
    }

    template<typename Repr>
    static auto compose(const Identity&, const Repr& Y)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      return Y;
    }

    static auto compose(const Identity&, const Identity&)
    {
      return Identity{};
    }

    template<typename Repr>
    static auto inverse(const Repr& X)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      return Derived::inverseImpl(X);
    }

    static auto inverse(const Identity& X)
    {
      return X;
    }

    template<typename Repr>
    static auto adjoint(const Repr& X)
    {
      static_assert(std::is_convertible_v<Repr, repr_t>);
      if constexpr (std::is_same_v<Repr, repr_t>)
        return Adjoint<Derived, ReprRef<Derived>>(X);
      else
        return Adjoint<Derived>(repr_t(X));
    }

    static auto adjoint(repr_t&& X)
    {
      return Adjoint<Derived>(std::forward<repr_t>(X));
    }

    static auto adjoint(const Identity& X)
    {
      return Adjoint(X);
    }

    template<typename Tan>
    static auto rightJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::jacobianImpl(t, rOnly);
    }

    template<typename Tan>
    static auto leftJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::jacobianImpl(t, lOnly);
    }

    template<typename Tan>
    static auto jacobians(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::jacobianImpl(t, both);
    }

    template<typename Tan>
    static auto invRightJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::invJacobianImpl(t, rOnly);
    }

    template<typename Tan>
    static auto invLeftJacobian(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::invJacobianImpl(t, lOnly);
    }

    template<typename Tan>
    static auto invJacobians(const Tan& t)
    {
      static_assert(std::is_convertible_v<Tan, tan_t> || alsoAcceptAsTan<Tan>());
      return Derived::invJacobianImpl(t, both);
    }

  private:
    template<typename Repr>
    static constexpr bool alsoAcceptAsRepr() { return std::is_same_v<Repr, Identity>; }

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