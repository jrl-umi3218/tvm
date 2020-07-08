/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/LieGroup.h>

namespace tvm::manifold::operators
{
  /** Given a Lie group LG and two elements X, Y of it, compute the composition X o Y*/
  template<typename LG>
  class Plus
  {
  public:
    using group_t = LG;
    using out_group_t = LG;
    using repr_t = typename LG::repr_t;
    using ret_t = typename LG::repr_t;
    template<typename T>
    static constexpr bool isLocal()
    {
      static_assert(std::is_same_v<T, internal::right_t> || std::is_same_v<T, internal::left_t>, "Invalid type: righ_t or left_t expected.");
      return std::is_same_v<T, internal::right_t>;
    }

    template<typename LHS, typename RHS>
    static auto value(const LHS& X, const RHS& Y)
    {
      return LG::compose(X, Y);
    }

    template<typename LHS, typename RHS, typename FromSide = internal::right_t, typename ToSide = FromSide>
    static auto jacobians(const LHS& X, const RHS& Y, FromSide = {}, ToSide = {})
    {
      // Z = X o Y
      if constexpr (isLocal<ToSide>())
      {
        if constexpr (isLocal<FromSide>())
        {
          // ^{Z}DZ/^{X}DX, ^{Z}DZ/^{Y}DY
          int dim = LG::dynamicDim(Y);
          return std::make_pair(LG::adjoint(Y).inverse().matrix(), typename LG::jacobian_t::Identity(dim, dim));
        }
        else
        {
          // ^{Z}DZ/^{E}DX, ^{Z}DZ/^{E}DY
          return std::make_pair((LG::adjoint(Y).inverse()*LG::adjoint(X).inverse()).matrix(), LG::adjoint(Y).inverse().matrix());
        }
      }
      else
      {
        if constexpr (isLocal<FromSide>())
        {
          // ^{E}DZ/^{X}DX, ^{E}DZ/^{Y}DY
          int dim = LG::dynamicDim(Y);
          return std::make_pair(LG::adjoint(X).matrix(), (LG::adjoint(X)* LG::adjoint(Y)).matrix());
        }
        else
        {
          // ^{E}DZ/^{E}DX, ^{E}DZ/^{E}DY
          int dim = LG::dynamicDim(X);
          return std::make_pair(typename LG::jacobian_t::Identity(dim, dim), LG::adjoint(X).matrix());
        }
      }
    }
  };
}