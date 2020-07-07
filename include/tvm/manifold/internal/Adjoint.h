/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <type_traits>

#include <tvm/internal/meta.h>

TVM_CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(LG)

namespace tvm::manifold::internal
{
  template<typename LG>
  struct ReprRef
  {
    using type = typename LG::repr_t;
    const type& X_;

    ReprRef(const type& X) : X_(X) {}
  };

  template<typename LG>
  struct ReprOwn
  {
    using type = typename LG::repr_t;
    type X_;

    ReprOwn(const type& X) : X_(X) {}
  };

  template<typename T, typename = void>
  struct has_X_member : std::false_type {};

  template<typename T>
  struct has_X_member<T, std::void_t<decltype(std::declval<T>().X_)>> : std::true_type {};

  template<typename T> 
  static const auto& get_repr(const T& t) 
  { 
    if constexpr (has_X_member<T>::value)
      return t.X_;
    else
      return t;
  }

  template<typename LG>
  struct AdjointOperations
  {
  };


  template<typename LG_, typename Expr_=ReprOwn<LG_>, bool Inverse_=false, bool Transpose_=false, bool PositiveSign_=true>
  class Adjoint
  {
  public:
    using LG = LG_;
    using Expr = Expr_;
    static constexpr bool Inverse = Inverse_;
    static constexpr bool Transpose = Transpose_;
    static constexpr bool PositiveSign = PositiveSign_;
    static constexpr int dim = LG::dim;
    using matrix_t = typename AdjointOperations<LG>::matrix_t;
    template<typename T>
    using isId = std::is_same<T, typename LG::Identity>;
    static constexpr bool exprIsId = isId<Expr>::value;


    explicit Adjoint(const Expr& X) : operand_(X) {}

    const auto& operand() const { return get_repr(operand_); }

    template<bool U = Inverse_, typename = std::enable_if_t<U, int> >
    auto reducedOperand() const { return LG::template inverse(operand()); }

    template<bool U = Inverse_, typename = std::enable_if_t<!U, int> >
    const auto& reducedOperand() const { return operand(); }

    template<typename U = Expr_, typename = std::enable_if_t<(isId<U>::value && dim < 0), int> >
    auto matrix(typename matrix_t::Index s) const
    {
      static_assert(std::is_same_v<U, Expr_>);
      return matrix_t::Identity(s,s);
    }

    template<typename U = Expr_, typename = std::enable_if_t<!(isId<U>::value && dim < 0), int> >
    auto matrix() const
    {
      static_assert(std::is_same_v<U, Expr_>);
      if constexpr (exprIsId)
      {
        if constexpr (dim >= 0)
          return matrix_t::Identity();
        else
          static_assert(tvm::internal::always_false<Expr>::value, "we should not reach here");
      }
      else
      {
        // We're getting a temporary here. When using it with - or transpose(), we need to force
        // the evaluation. Otherwise we would return an Eigen expression based on a temporary.
        const auto& mat = AdjointOperations<LG>::toMatrix(reducedOperand());
        if constexpr (Transpose)
        {
          if constexpr (PositiveSign)
            return matrix_t(mat.transpose());
          else
            return matrix_t(-mat.transpose());
        }
        else
        {
          if constexpr (PositiveSign)
            return mat;
          else
            return matrix_t(-mat);
        }
      }
    }

    auto inverse() const
    {
      return Adjoint<LG, Expr, !Inverse, Transpose, PositiveSign>(operand_);
    }

    auto transpose() const
    {
      return Adjoint<LG, Expr, Inverse, !Transpose, PositiveSign>(operand_);
    }

    auto dual() const
    {
      return Adjoint<LG, Expr, !Inverse, !Transpose, PositiveSign>(operand_);
    }

    auto operator-() const
    {
      return Adjoint<LG, Expr, Inverse, Transpose, !PositiveSign>(operand_);
    }

    template<typename OtherExpr, bool OtherInverse, bool OtherTranspose, bool OtherPositiveSign>
    auto operator*(const Adjoint<LG, OtherExpr, OtherInverse, OtherTranspose, OtherPositiveSign>& other)
    {
      using Id = typename LG::Identity;
      constexpr bool otherExprIsId = isId<OtherExpr>::value;
      constexpr bool retSign = PositiveSign == OtherPositiveSign;
      if constexpr (exprIsId)
      {
        if constexpr (otherExprIsId)
          return Adjoint<LG, Id, false, false, retSign>(Id{});
        else
          return Adjoint<LG, OtherExpr, OtherInverse, OtherTranspose, retSign>(other.operand());
      }
      else
      {
        if constexpr (otherExprIsId)
          return Adjoint<LG, Expr, Inverse, Transpose, retSign>(operand_);
        else
        {
          static_assert(!Transpose && !OtherTranspose, "Cannot use transpose types here. Pass to the matrix representation before performing the multiplication.");
          using ReprP = ReprOwn<LG>;
          if constexpr (Inverse)
          {
            if constexpr (OtherInverse) //inv(Ad_x)*inv(Ad_y) = inv(Ad_{y*x})
              return Adjoint<LG, ReprP, true, false, retSign>(ReprP{ LG::template compose(other.operand(), this->operand()) });
            else //inv(Ad_x)*Ad_y = Ad_{inv(x)*y}
              return Adjoint<LG, ReprOwn<LG>, false, false, retSign>(ReprP{ LG::template compose(this->reducedOperand(), other.operand()) });
          }
          else
          {
            if constexpr (OtherInverse) //Ad_x*inv(Ad_y) = Ad_{x*inv(y)}
              return Adjoint<LG, ReprP, false, false, retSign>(ReprP{ LG::template compose(this->operand(), other.reducedOperand()) });
            else //Ad_x*Ad_y = Ad_{x*y}
              return Adjoint<LG, ReprP, false, false, retSign>(ReprP{ LG::template compose(this->operand(), other.operand()) });
          }
        }
      }
    }

  private:
    Expr operand_;
  };

  // Deduction guide for identity
  template<typename Id>
  Adjoint(const Id& id)->Adjoint<typename Id::LG, Id>;
}