/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <type_traits>

#include <tvm/internal/meta.h>

TVM_CREATE_HAS_MEMBER_TYPE_TRAIT_FOR(LG)

namespace tvm::manifold::internal
{
  /** Small wrapper around a const reference on an object of type LG::repr_t*/
  template<typename LG>
  struct ReprRef
  {
    using type = typename LG::repr_t;
    const type& X_;

    ReprRef(const type& X) : X_(X) {}
  };

  /** Small wrapper owning an object of type LG::repr_t*/
  template<typename LG>
  struct ReprOwn
  {
    using type = typename LG::repr_t;
    type X_;

    ReprOwn(const type& X) : X_(X) {}
  };

  /** Utility to detect that an object as a public member X_.*/
  template<typename T> using has_X_trait_t = decltype(std::declval<T>().X_);
  template<typename T> using has_X_member = tvm::internal::is_detected<has_X_trait_t, T>;

  /** Dispatch utility: it \p t.X_ is a valid expression, return its value, otherwise return \p t.*/
  template<typename T> 
  static const auto& get_repr(const T& t) 
  { 
    if constexpr (has_X_member<T>::value)
      return t.X_;
    else
      return t;
  }

  /** Struct to be specialized if some operations need to be specified for a given Lie group.*/
  template<typename LG>
  struct AdjointOperations {};

  /** Class representing the Adjoint of an element of the Lie group \p LG_.
    *
    * \tparam LG_ The Lie Group
    * \tparam Expr_ The type of the underlying expression representing an element of the Lie group.
    * \internal At the present, the only type used are ReprOwn<LG_>, ReprRef<LG_> and LG_::Identity
    * but we could imagine having Eigen-like expressions for elements of the group.
    * \tparam Inverse_ Whether this is the inverse of the adjoint of the underlying element or not
    * \tparam Transpose_ Whether the adjoint (matrix) should be transpose or not. In conjunction
    * with Inverse_, we can express the dual.
    * \tparam PositiveSign_ If false, we have -Ad_X.
    *
    * \internal This design is an attempt at avoiding CRTP and numerous classes, while allowing to
    * have "non-owning" object through the ReprRef mechanism to avoid copies, and not explicitely
    * computing the inverse of the underlying element or the dual of the adoint when its not
    * necessary. It's not fully clean though, and there are room for improvements, that would be
    * easier to carry out with a full Eigen-like expression system on the Lie group elements. 
    */
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

    /** Get the underlying element. */
    const auto& operand() const { return get_repr(operand_); }

    /** Get the operand or its inverse, depending on the Inverse_ flag*/
    template<bool U = Inverse_, typename = std::enable_if_t<U, int> >
    auto reducedOperand() const { return LG::template inverse(operand()); }

    /** Get the operand or its inverse, depending on the Inverse_ flag*/
    template<bool U = Inverse_, typename = std::enable_if_t<!U, int> >
    const auto& reducedOperand() const { return operand(); }

    /** Get the matrix representation of the adjoint.
      * This overload is meant for the special case when the underlying element is
      * the identiy and the manifold has a dynamic size.
      */
    template<typename U = Expr_, typename = std::enable_if_t<(isId<U>::value && !LG::hasStaticDim()), int> >
    auto matrix(typename matrix_t::Index s) const
    {
      static_assert(std::is_same_v<U, Expr_>);
      return matrix_t::Identity(s,s);
    }

    /** Get the matrix representation of the adjoint.*/
    template<typename U = Expr_, typename = std::enable_if_t<!(isId<U>::value && !LG::hasStaticDim()), int> >
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

    /** Return the inverse of the adjoint.*/
    auto inverse() const
    {
      return Adjoint<LG, Expr, !Inverse, Transpose, PositiveSign>(operand_);
    }

    /** Return the transpose of the adjoint.
      *
      * \internal Mathematically speaking, the transpose of the adjoint itself is
      * not defined AFAIK. This concerns the matrix representation of the adjoint
      * and comes into play when the matrix multiplies another element, so maybe
      * the transpose could be defered to the multiplication operation.
      */
    auto transpose() const
    {
      return Adjoint<LG, Expr, Inverse, !Transpose, PositiveSign>(operand_);
    }

    /** Return the dual of the adjoint.*/
    auto dual() const
    {
      return Adjoint<LG, Expr, !Inverse, !Transpose, PositiveSign>(operand_);
    }

    /** Return the opposite of the adjoint.
      *
      * \internal Again, not mathematically perfect. We mix the adjoit and its
      * matrix representation here.
      */
    auto operator-() const
    {
      return Adjoint<LG, Expr, Inverse, Transpose, !PositiveSign>(operand_);
    }

    /** Multiplication between adjoints.*/
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
    Expr operand_;  //The underlying element of LG.
  };

  // Deduction guide for identity
  template<typename Id>
  Adjoint(const Id& id)->Adjoint<typename Id::LG, Id>;
}