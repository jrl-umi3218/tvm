/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/Manifold.h>

namespace tvm::manifold::internal
{
  // Forward declaration
  template<typename Derived> class LieGroup;

  /** Specialization of traits for LieGroup*/
  template<typename Derived>
  struct traits<LieGroup<Derived>>
  {
    static constexpr int dim = traits<Derived>::dim;
    using repr_t = typename traits<Derived>::repr_t;
  };

  /** This class serves as a base class for all Lie groups.
    * It introduces the composition operation, the inverse
    * and the identity.
    */
  template<typename Derived>
  class LieGroup : public Manifold<LieGroup<Derived>>
  {
  public:
    /** Identity of the group*/
    struct Identity {};
    /** Identity of the algebra*/
    struct AlgIdentity {};

    static constexpr auto identity = Identity{};
    static constexpr auto algIdentity = AlgIdentity{};

    using Base = Manifold<LieGroup<Derived>>;
    using typename Base::repr_t;
    using typename Base::tan_t;

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