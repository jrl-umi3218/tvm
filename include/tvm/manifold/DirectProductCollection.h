/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/manifold/DirectProduct.h>
#include <tvm/manifold/Real.h>
#include <tvm/manifold/SO3.h>

namespace tvm::manifold
{
  namespace internal
  {
    /** A specialization for product of vector spaces */
    template<int N1, int N2>
    struct product_traits<Real<N1>, Real<N2>, defaut_direct_product_repr_t<Real<N1>, Real<N2>>>
    {
      using LG1 = Real<N1>;
      using LG2 = Real<N2>;
      using Repr = defaut_direct_product_repr_t<Real<N1>, Real<N2>>;
      using Prod = DirectProduct<LG1, LG2, Repr>;
      using tan1_t = typename LG1::tan_t;
      using tan2_t = typename LG2::tan_t;

      template <typename R>
      static auto elem1(const R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG1::hasStaticDim()) return X.template head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return X.head(X.size() - LG2::dim);
        else static_assert(tvm::internal::always_false<R>::value, "Unable to determine the size of elem1.");
      }

      template <typename R>
      static auto elem1(R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG1::hasStaticDim()) return X.template head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return X.head(X.size() - LG2::dim);
        else static_assert(tvm::internal::always_false<R>::value, "Unable to determine the size of elem1.");
      }

      template <typename R>
      static auto elem2(const R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG2::hasStaticDim()) return X.template tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return X.tail(X.size() - LG1::dim);
        else static_assert(tvm::internal::always_false<R>::value, "Unable to determine the size of elem2.");
      }

      template <typename R>
      static auto elem2(R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (LG2::hasStaticDim()) return X.template tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return X.tail(X.size() - LG1::dim);
        else static_assert(tvm::internal::always_false<R>::value, "Unable to determine the size of elem2.");
      }

      template <typename Tan>
      static auto tan1(const Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (LG1::hasStaticDim()) return t.template head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return t.head(t.size() - LG2::dim);
        else static_assert(tvm::internal::always_false<Tan>::value, "Unable to determine the size of tan1.");
      }

      template <typename Tan>
      static auto tan1(Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (LG1::hasStaticDim()) return t.template head<LG1::dim>();
        else if constexpr (LG2::hasStaticDim()) return t.head(t.size() - LG2::dim);
        else static_assert(tvm::internal::always_false<Tan>::value, "Unable to determine the size of tan1.");
      }

      template <typename Tan>
      static auto tan2(const Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan2_t>);
        if constexpr (LG2::hasStaticDim()) return t.template tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return t.tail(t.size() - LG1::dim);
        else static_assert(tvm::internal::always_false<Tan>::value, "Unable to determine the size of tan2.");
      }

      template <typename Tan>
      static auto tan2(Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan2_t>);
        if constexpr (LG2::hasStaticDim()) return t.template tail<LG2::dim>();
        else if constexpr (LG1::hasStaticDim()) return t.tail(t.size() - LG1::dim);
        else static_assert(tvm::internal::always_false<Tan>::value, "Unable to determine the size of tan2.");
      }
    };


    template<typename Repr>
    struct product_traits<SO3, Real<3>, Repr>
    {
      using R3 = Real<3>;
      using Prod = DirectProduct<SO3, R3, Repr>;
      using tan1_t = typename SO3::tan_t;
      using tan2_t = typename R3::tan_t;
      static constexpr bool is_pair_v = std::is_same_v<Repr, std::pair<Eigen::Matrix3d, Eigen::Vector3d>>;

      template <typename R>
      static const auto& elem1(const R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (is_pair_v) return X.first;
        else return X.rotation();
      }

      template <typename R>
      static auto& elem1(R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (is_pair_v) return X.first;
        else return X.rotation();
      }

      template <typename R>
      static const auto& elem2(const R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (is_pair_v) return X.second;
        else return X.translation();
      }

      template <typename R>
      static auto& elem2(R& X)
      {
        static_assert(std::is_convertible_v<R, Repr>);
        if constexpr (is_pair_v) return X.second;
        else return X.translation();
      }

      template <typename Tan>
      static auto tan1(const Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (is_pair_v) return t.template head<3>();
        else return std::cref(t.angular());
      }

      template <typename Tan>
      static auto tan1(Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (is_pair_v) return t.template head<3>();
        else return std::ref(t.angular());
      }

      template <typename Tan>
      static auto tan2(const Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (is_pair_v) return t.template tail<3>();
        else return std::cref(t.linear());
      }

      template <typename Tan>
      static auto tan2(Tan& t)
      {
        static_assert(std::is_convertible_v<Tan, tan1_t>);
        if constexpr (is_pair_v) return t.template tail<3>();
        else return std::ref(t.linear());
      }
    };
  }
}