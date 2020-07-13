/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/internal/Adjoint.h>
#include <tvm/manifold/internal/LieGroup.h>

namespace tvm::manifold
{
  template<typename LG1, typename LG2, typename Repr> class DirectProduct;

  namespace internal
  {

    template<typename LG1, typename LG2>
    inline constexpr int productDim()
    {
      if constexpr (LG1::hasStaticDim())
      {
        if constexpr (LG2::hasStaticDim())
          return LG1::dim + LG2::dim;
        else
          return LG2::dim;
      }
      else
        return LG1::dim;
    }

    template<typename LG1, typename LG2>
    using defaut_direct_product_repr_t = typename Eigen::Matrix<double, internal::productDim<LG1, LG2>(), 1>;

    /** A struct to be specialized for different direct products of Lie groups.
      * The default implementation is for vector spaces (products of Rk).
      *
      * \internal This could be made more generic by having a notion of size of the representation element.
      */
    template<typename LG1, typename LG2, typename Repr = typename defaut_direct_product_repr_t<LG1, LG2>>
    struct product_traits
    {
      using Prod = DirectProduct<LG1, LG2, Repr>;
      using tan1_t = typename LG1::tan_t;
      using tan2_t = typename LG2::tan_t;

      template <typename R>
      static auto elem1(const R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG1::hasStaticDim()) return X.head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return X.head(X.size() - LG2::dim);
        else static_assert(false, "Unable to determine the size of elem1.");
      }

      template <typename R>
      static auto elem1(R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr(LG1::hasStaticDim()) return X.head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return X.head(X.size() - LG2::dim);
        else static_assert(false, "Unable to determine the size of elem1.");
      }

      template <typename R>
      static auto elem2(const R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG2::hasStaticDim()) return X.tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return X.tail(X.size() - LG1::dim);
        else static_assert(false, "Unable to determine the size of elem2.");
      }

      template <typename R>
      static auto elem2(R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG2::hasStaticDim()) return X.tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return X.tail(X.size() - LG1::dim);
        else static_assert(false, "Unable to determine the size of elem2.");
      }

      template <typename Tan>
      static auto tan1(const Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (LG1::hasStaticDim()) return t.head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return t.head(t.size() - LG2::dim);
        else static_assert(false, "Unable to determine the size of tan1.");
      }

      template <typename Tan>
      static auto tan1(Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (LG1::hasStaticDim()) return t.head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return t.head(t.size() - LG2::dim);
        else static_assert(false, "Unable to determine the size of tan1.");
      }

      template <typename Tan>
      static auto tan2(const Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan2_t>);
        if constexpr (LG2::hasStaticDim()) return t.tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return t.tail(t.size() - LG1::dim);
        else static_assert(false, "Unable to determine the size of tan2.");
      }

      template <typename Tan>
      static auto tan2(Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan2_t>);
        if constexpr (LG2::hasStaticDim()) return t.tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return t.tail(t.size() - LG1::dim);
        else static_assert(false, "Unable to determine the size of tan2.");
      }
    };


    template<typename LG1, typename LG2, typename Repr>
    struct traits<DirectProduct<LG1, LG2, Repr>>
    {
      static constexpr int dim = productDim<LG1, LG2>();
      using repr_t = Repr;
      template<typename R>
      static int dynamicDim(const R& X)
      {
        if constexpr (dim >= 0)
          return dim;
        else
          return LG1::dynamicDim(product_traits<LG1, LG2, Repr>::elem1(X))
               + LG2::dynamicDim(product_traits<LG1, LG2, Repr>::elem2(X));
      }
    };


    template<typename LG1, typename LG2, typename Repr>
    struct AdjointOperations<DirectProduct<LG1, LG2, Repr>>
    {
      using Prod = DirectProduct<LG1, LG2, Repr>;
      using matrix_t = Eigen::Matrix<double, Prod::dim, Prod::dim>;
      static auto toMatrix(const typename Prod::repr_t& X) 
      { 
        return matrix_t::Identity(X.size(), X.size()); 
      }
    };
  }


  template<typename LG1, typename LG2, typename Repr = typename internal::defaut_direct_product_repr_t<LG1, LG2>>
  class DirectProduct: public internal::LieGroup<DirectProduct<LG1, LG2, Repr>>
  {
  public:
    using Base = internal::LieGroup<DirectProduct<LG1, LG2, Repr>>;
    using pt = internal::product_traits<LG1, LG2, Repr>;
    using typename Base::repr_t;
    using typename Base::tan_t;
    using typename Base::jacobian_t;

  private:
    template<typename Repr>
    static const auto& logImpl(const Repr& X)
    {
      tan_t ret;
      pt::tan1(ret) = LG1::log(pt::elem1(X));
      pt::tan2(ret) = LG2::log(pt::elem2(X));
      return ret;
    }

    template<typename Tan>
    static const auto& expImpl(const Tan& t)
    {
      repr_t ret;
      pt::elem1(ret) = LG1::exp(pt::tan1(t));
      pt::elem2(ret) = LG2::exp(pt::tan2(t));
      return ret;
    }

    template<typename ReprX, typename ReprY>
    static auto composeImpl(const ReprX& X, const ReprY& Y)
    {
      repr_t ret;
      pt::elem1(ret) = LG1::compose(pt::elem1(X), pt::elem1(Y));
      pt::elem2(ret) = LG2::compose(pt::elem2(X), pt::elem2(Y));
      return ret;
    }

    template<typename Repr>
    static auto inverseImpl(const Repr& X)
    {
      repr_t ret;
      pt::elem1(ret) = LG1::inverse(pt::elem1(X));
      pt::elem2(ret) = LG2::inverse(pt::elem2(X));
      return ret;
    }

  //  template<typename Tan, typename Which>
  //  static auto jacobianImpl(const Tan& t, Which)
  //  {
  //    if constexpr (std::is_same_v<Which, internal::right_t> || std::is_same_v<Which, internal::left_t>)
  //      return jacobian_t::Identity(t.size(), t.size());
  //    else
  //    {
  //      static_assert(std::is_same_v<Which, internal::both_t>);
  //      return std::make_pair(jacobian_t::Identity(t.size(), t.size()), jacobian_t::Identity(t.size(), t.size()));
  //    }
  //  }

  //  template<typename Tan, typename Which>
  //  static auto invJacobianImpl(const Tan& t, Which)
  //  {
  //    if constexpr (std::is_same_v<Which, internal::right_t> || std::is_same_v<Which, internal::left_t>)
  //      return jacobian_t::Identity(t.size(), t.size());
  //    else
  //    {
  //      static_assert(std::is_same_v<Which, internal::both_t>);
  //      return std::make_pair(jacobian_t::Identity(t.size(), t.size()), jacobian_t::Identity(t.size(), t.size()));
  //    }
  //  }

    friend Base;
  };
}