/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */
#pragma once


namespace tvm::internal
{
  /** Constexpr integer power base^exp
    * Adapted from https://stackoverflow.com/a/17728525
    */
  template <typename T>
  constexpr T pow(T base, unsigned int exp, T result = 1)
  {
    return exp <= 1 ? (exp == 0 ? 1 : result * base) : pow(base * base, exp / 2, (exp % 2) ? result * base : result);
  }

  /* Constexpr version of the square root of x
    * curr is the initial guess for the square root
    * Adapted from https://gist.github.com/alexshtf/eb5128b3e3e143187794
    */
  constexpr double sqrtNewtonRaphson(double x, double curr, double prev = 0)
  {
    return curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
  }

  /**
   * Constexpr version of the square root
   * Return value:
   *   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
   *   - Otherwise, returns NaN
   * Copied from https://stackoverflow.com/a/34134071
   */
  template<typename T>
  T constexpr sqrt(T x)
  {
    return x >= static_cast<T>(0) && x < std::numeric_limits<T>::infinity() ? sqrtNewtonRaphson(x, x, static_cast<T>(0))
      : std::numeric_limits<T>::quiet_NaN();
  }
}