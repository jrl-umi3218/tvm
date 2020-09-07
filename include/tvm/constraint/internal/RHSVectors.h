/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <tvm/api.h>
#include <tvm/constraint/enums.h>

#include <Eigen/Core>

namespace tvm
{

namespace constraint
{

namespace internal
{

/** This class manages the vectors \p l, \p u and \p e that appear in the
 * various types of constraints.
 */
class RHSVectors
{
public:
  /** Create a instance for the given conventions
   * \param ct The constraint type
   * \param cr The type of "right hand side"
   */
  RHSVectors(Type ct, RHS cr);

  /** Resize the vectors used to size \p n.*/
  void resize(int n);

  /** Return vector \p l
   * \attention This does not check if \p l is in use
   */
  Eigen::VectorXd & l();

  /** Return vector \p l (const version)
   * \attention This does not check if \p l is in use
   */
  const Eigen::VectorXd & l() const;

  /** Return vector \p u
   * \attention This does not check if \p u is in use
   */
  Eigen::VectorXd & u();

  /** Return vector \p  u(const version)
   * \attention This does not check if \p u is in use
   */
  const Eigen::VectorXd & u() const;

  /** Return vector \p e
   * \attention This does not check if \p e is in use
   */
  Eigen::VectorXd & e();

  /** Return vector \p e (const version)
   * \attention This does not check if \p e is in use
   */
  const Eigen::VectorXd & e() const;

  /** Return the vector specified by t:
   * - l if t == Type::GREATER_THAN
   * - u if t == Type::LOWER_THAN
   * - e if t == Type::EQUAL
   * If t == Type::DOUBLE_SIDED, throw an exception
   */
  Eigen::VectorXd & rhs(Type t);

  /** Return the vector specified by t (const version):
   * - l if t == Type::GREATER_THAN
   * - u if t == Type::LOWER_THAN
   * - e if t == Type::EQUAL
   * If t == Type::DOUBLE_SIDED, throw an exception
   */
  const Eigen::VectorXd & rhs(Type t) const;

  /** Check if \p l is used.*/
  bool use_l() const;

  /** Check if \p u is used.*/
  bool use_u() const;

  /** Check if \p e is used.*/
  bool use_e() const;

private:
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;
  Eigen::VectorXd e_;

  const bool use_l_;
  const bool use_u_;
  const bool use_e_;
};

inline Eigen::VectorXd & RHSVectors::l() { return l_; }

inline const Eigen::VectorXd & RHSVectors::l() const { return l_; }

inline Eigen::VectorXd & RHSVectors::u() { return u_; }

inline const Eigen::VectorXd & RHSVectors::u() const { return u_; }

inline Eigen::VectorXd & RHSVectors::e() { return e_; }

inline const Eigen::VectorXd & RHSVectors::e() const { return e_; }

inline Eigen::VectorXd & RHSVectors::rhs(Type t)
{
  switch(t)
  {
    case Type::LOWER_THAN:
      return u_;
    case Type::GREATER_THAN:
      return l_;
    case Type::EQUAL:
      return e_;
    case Type::DOUBLE_SIDED:
      throw std::runtime_error("This methods is not available for DOUBLE_SIDED");
  }
}

inline const Eigen::VectorXd & RHSVectors::rhs(Type t) const
{
  switch(t)
  {
    case Type::LOWER_THAN:
      return u_;
    case Type::GREATER_THAN:
      return l_;
    case Type::EQUAL:
      return e_;
    case Type::DOUBLE_SIDED:
      throw std::runtime_error("This methods is not available for DOUBLE_SIDED");
  }
}

inline bool RHSVectors::use_l() const { return use_l_; }

inline bool RHSVectors::use_u() const { return use_u_; }

inline bool RHSVectors::use_e() const { return use_e_; }

} // namespace internal

} // namespace constraint

} // namespace tvm