/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/LieGroup.h>
#include <tvm/manifold/Real.h>

namespace tvm::manifold::operators
{
  /** Given a Lie group LG and two elements X, Y of it, compute  log(X^-1 o Y)*/
  template<typename LG>
  class RMinus
  {
  public:
    using group_t = LG;
    using out_group_t = tvm::manifold::Real<LG::dim>;
    using repr_t = typename group_t::repr_t;
    using ret_t = typename out_group_t::repr_t;

    template<typename LHS, typename RHS>
    static auto value(const LHS& X, const RHS& Y)
    {
      return LG::log(LG::compose(LG::inverse(X), Y));
    }

    template<typename LHS, typename RHS, typename FromSide = internal::right_t, typename ToSide = FromSide>
    static auto jacobians(const LHS& X, const RHS& Y, FromSide = {}, ToSide = {})
    {
      //We cast all outputs to matrix_t to ensure we have no expression with references
      //This is sub-optimal in several cases.
      using matrix_t = typename out_group_t::jacobian_t; 
      // t = log(Z) with Z = X^-1 o Y
      ret_t t = value(X, Y);
      if constexpr (internal::isLocal<FromSide>())
      {
        // Dt/^{X}DX, Dt/^{Y}DY
        const auto& [invJr, invJl] = LG::invJacobians(t);
        return std::make_pair(matrix_t(-invJl), invJr);
      }
      else
      {
        // Dt/^{E}DX, Dt/^{E}DY
        const auto& [invJr, invJl] = LG::invJacobians(t);
        return std::make_pair(matrix_t(-invJl*LG::adjoint(X).inverse().matrix()), matrix_t(invJr* LG::adjoint(Y).inverse().matrix()));
      }
    }
  };
}