/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once

#include <tvm/internal/math.h>

namespace tvm::manifold::internal
{

  /** sinus cardinal: sin(x)/x
    * Code adapted from boost::math::detail::sinc
    */
  template<typename T>
  T sinc(const T x)
  {
    constexpr T taylor_0_bound = std::numeric_limits<double>::epsilon();
    constexpr T taylor_2_bound = tvm::internal::sqrt(taylor_0_bound);
    constexpr T taylor_n_bound = tvm::internal::sqrt(taylor_2_bound);

    if (std::abs(x) >= taylor_n_bound)
    {
      return (std::sin(x) / x);
    }
    else
    {
      // approximation by taylor series in x at 0 up to order 0
      T result = static_cast<T>(1);

      if (std::abs(x) >= taylor_0_bound)
      {
        T x2 = x * x;

        // approximation by taylor series in x at 0 up to order 2
        result -= x2 / static_cast<T>(6);

        if (std::abs(x) >= taylor_2_bound)
        {
          // approximation by taylor series in x at 0 up to order 4
          result += (x2 * x2) / static_cast<T>(120);
        }
      }

      return (result);
    }
  }

  /** sinus cardinal when x^2 is available: sin(sqrt(x))/sqrt(x)
    * Code adapted from boost::math::detail::sinc
    */
  template<typename T>
  T sincsqrt(const T x2)
  {
    assert(x2 >= 0);
    constexpr T taylor_0_bound = std::numeric_limits<double>::epsilon();
    constexpr T taylor_s_bound = taylor_0_bound * taylor_0_bound;
    constexpr T taylor_2_bound = tvm::internal::sqrt(taylor_0_bound);

    if (x2 >= taylor_2_bound)
    {
      auto x = sqrt(x2);
      return (std::sin(x) / x);
    }
    else
    {
      // approximation by taylor series in x at 0 up to order 0
      T result = static_cast<T>(1);

      if (x2 >= taylor_s_bound)
      {
        // approximation by taylor series in x at 0 up to order 2
        result -= x2 / static_cast<T>(6);

        if (x2 >= taylor_0_bound)
        {
          // approximation by taylor series in x at 0 up to order 4
          result += (x2 * x2) / static_cast<T>(120);
        }
      }

      return (result);
    }
  }

  /**
   * Compute 1/sinc(x).
   * This code is inspired by boost/math/special_functions/sinc.hpp.
   */
  template<typename T>
  T sinc_inv(const T x)
  {
    constexpr T taylor_0_bound = std::numeric_limits<T>::epsilon();
    constexpr T taylor_2_bound = tvm::internal::sqrt(taylor_0_bound);
    constexpr T taylor_n_bound = tvm::internal::sqrt(taylor_2_bound);

    // We use the 4th order taylor series around 0 of x/sin(x) to compute
    // this function:
    //      2      4
    //     x    7⋅x     ⎛ 6⎞
    // 1 + ── + ──── + O⎝x ⎠
    //     6    360
    // this approximation is valid around 0.
    // if x is far from 0, our approximation is not valid
    // since x^6 becomes non negligable we use the normal computation of the function
    // (i.e. taylor_2_bound^6 + taylor_0_bound == taylor_0_bound but
    //       taylor_n_bound^6 + taylor_0_bound != taylor_0).

    if (std::abs(x) >= taylor_n_bound)
    {
      return (x / std::sin(x));
    }
    else
    {
      // x is bellow taylor_n_bound so we don't care of the 6th order term of
      // the taylor series.
      // We set the 0 order term.
      T result = static_cast<T>(1);

      if (std::abs(x) >= taylor_0_bound)
      {
        // x is above the machine epsilon so x^2 is meaningful.
        T x2 = x * x;
        result += x2 / static_cast<T>(6);

        if (std::abs(x) >= taylor_2_bound)
        {
          // x is upper the machine sqrt(epsilon) so x^4 is meaningful.
          result += static_cast<T>(7) * (x2 * x2) / static_cast<T>(360);
        }
      }

      return (result);
    }
  }

  /// Compute the value \f$ \frac{1}{x^2} - \frac{1+\cos(x)}{2 x \sin(x)} \f$.
  template <typename T>
  inline T SO3JacInvF2(const T& x)
  {
    using tvm::internal::sqrt;
    constexpr T ulp = std::numeric_limits<T>::epsilon();
    constexpr T taylor_0_bound = ulp;
    constexpr T taylor_2_bound = sqrt(60 * ulp);
    constexpr T taylor_4_bound = sqrt(sqrt(2520 * ulp));
    constexpr T taylor_8_bound = sqrt(sqrt(sqrt(3991680 * ulp)));

    double absx = std::abs(x);
    if (absx >= taylor_8_bound)
    {
      return static_cast<T>(1) / (x * x) - (static_cast<T>(1) + std::cos(x)) / (static_cast<T>(2) * x * std::sin(x));
    }
    else
    {
      // approximation by taylor series in x at 0 up to order 0
      T result = static_cast<T>(1);

      if (absx >= taylor_0_bound)
      {
        T x2 = x * x;
        // approximation by taylor series in x at 0 up to order 2
        result += x2 / static_cast<T>(60);

        if (absx >= taylor_2_bound)
        {
          T x4 = x2 * x2;
          // approximation by taylor series in x at 0 up to order 4
          result += x4 / static_cast<T>(2520);

          if (absx >= taylor_4_bound)
          {
            // approximation by taylor series in x at 0 up to order 8
            result += (x2 * x4) / static_cast<T>(100800) + (x4 * x4) / static_cast<T>(3991680);
          }
        }
      }

      return result / static_cast<T>(12);
    }
  }

  template <typename T>
  inline std::pair<T,T> SO3JacF1F2(const T& x)
  {
    using tvm::internal::sqrt;
    constexpr T ulp = std::numeric_limits<T>::epsilon();
    constexpr T taylor_0_bound = ulp;
    constexpr T taylor_2_bound = sqrt(12 * ulp);
    constexpr T taylor_4_bound = sqrt(sqrt(360 * ulp));
    constexpr T taylor_8_bound = sqrt(sqrt(sqrt(1814400 * ulp)));

    double absx = std::abs(x);
    if (absx >= taylor_8_bound)
    {
      T x2 = x * x;
      return { (static_cast<T>(1) - std::cos(x)) / x2, (x - std::sin(x)) / (x2 * x) };
    }
    else
    {
      // approximation by taylor series in x at 0 up to order 0
      T f1 = static_cast<T>(1);
      T f2 = static_cast<T>(1);

      if (absx >= taylor_0_bound)
      {
        T x2 = x * x;
        // approximation by taylor series in x at 0 up to order 2
        f1 -= x2 / static_cast<T>(12);
        f2 -= x2 / static_cast<T>(20);

        if (absx >= taylor_2_bound)
        {
          T x4 = x2 * x2;
          // approximation by taylor series in x at 0 up to order 4
          f1 += x4 / static_cast<T>(360);
          f2 += x4 / static_cast<T>(840);

          if (absx >= taylor_4_bound)
          {
            // approximation by taylor series in x at 0 up to order 8
            T x6 = x2 * x4;
            T x8 = x4 * x4;
            f1 += x8 / static_cast<T>(1814400) - x6 / static_cast<T>(20160);
            f2 += x8 / static_cast<T>(6652800) - x6 / static_cast<T>(60480);
          }
        }
      }

      return { f1 / static_cast<T>(2), f2 / static_cast<T>(6) };
    }
  }
}