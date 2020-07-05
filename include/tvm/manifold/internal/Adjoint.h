/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <type_traits>

#include <tvm/internal/meta.h>

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


    explicit Adjoint(const Expr& X) : elem_(X) {}

    const auto& elem() const { return get_repr(elem_); }

    auto inverse() const
    {
      return Adjoint<LG, Expr, !Inverse, Transpose, PositiveSign>(elem_);
    }

    auto transpose() const
    {
      return Adjoint<LG, Expr, Inverse, !Transpose, PositiveSign>(elem_);
    }

    auto dual() const
    {
      return Adjoint<LG, Expr, !Inverse, !Transpose, PositiveSign>(elem_);
    }

    auto operator-() const
    {
      return Adjoint<LG, Expr, Inverse, Transpose, !PositiveSign>(elem_);
    }

    template<typename OtherExpr, bool OtherInverse, bool OtherTranspose, bool OtherPositiveSign>
    auto operator*(const Adjoint<LG, OtherExpr, OtherInverse, OtherTranspose, OtherPositiveSign>& other)
    {
      using Id = typename LG::Identity;
      constexpr bool exprIsId = std::is_same_v<Expr, Id>;
      constexpr bool otherExprIsId = std::is_same_v<OtherExpr, Id>;
      constexpr bool retSign = PositiveSign == OtherPositiveSign;
      if constexpr (exprIsId)
      {
        if constexpr (otherExprIsId)
          return Adjoint<LG, Id, false, false, retSign>(Id{});
        else
          return Adjoint<LG, OtherExpr, OtherInverse, OtherTranspose, retSign>(other.elem());
      }
      else
      {
        if constexpr (otherExprIsId)
          return Adjoint<LG, Expr, Inverse, Transpose, retSign>(elem_);
        else
        {
          static_assert(!Transpose && !OtherTranspose, "Cannot use transpose types here. Pass to the matrix representation before performing the multiplication.");
          if constexpr (Inverse)
          {
            if constexpr (OtherInverse) //inv(Ad_x)*inv(Ad_y) = inv(Ad_{y*x})
              return Adjoint<LG, ReprOwn<LG>, true, false, retSign>({ LG::template compose(other.elem(), this->elem()) });
            else //inv(Ad_x)*Ad_y = Ad_{inv(x)*y}
              return Adjoint<LG, ReprOwn<LG>, false, false, retSign>({ LG::template compose(LG::template inverse(this->elem()), other.elem()) });
          }
          else
          {
            if constexpr (OtherInverse) //Ad_x*inv(Ad_y) = Ad_{x*inv(y)}
              return Adjoint<LG, ReprOwn<LG>, false, false, retSign>({ LG::template compose(this->elem(), LG::template inverse(other.elem())) });
            else //Ad_x*Ad_y = Ad_{x*y}
              return Adjoint<LG, ReprOwn<LG>, false, false, retSign>({ LG::template compose(this->elem(), other.elem()) });
          }
        }
      }
    }

  private:
    Expr elem_;
  };
}