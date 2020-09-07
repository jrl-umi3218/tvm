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

#include <tvm/function/abstract/Function.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/internal/meta.h>

#include <algorithm>
#include <initializer_list>

namespace tvm
{

namespace utils
{

/** This class wraps a function to mask the update mechanism to the user, by
 * providing methods as value and jacobian that can be called directly with
 * the values of the variables.
 * It is meant for diagnostic and debugging purposes.
 */
class TVM_DLLAPI UpdatelessFunction
{
public:
  /** Constructor with the function to be wrapped. */
  UpdatelessFunction(FunctionPtr f);

  /** Get the value of the function for given values of its variables.
   * Variable values can be given as VectorXd, or as std::initializer_list.
   * There are three possible syntaxes:
   * - value(val1, val2, ..., valn) where val1, ..., valn are the values of
   *   the n variables of the function, in the order they are stored in the
   *   function (i.e. the order of the vector returned by variables() )
   * - value(xi1, vali1, xi2, vali2, ...) where one alternates a reference
   *   to one variable and its value. The number of variables in this case
   *   need not be n (unspecified variables will keep their previous value),
   *   and if a variable is given multiple times, only the last value will
   *   be used
   * - value(val) where val is a concatenated value for all variables in the
   *   order they are stored in the function.
   */
  template<typename... Vals>
  const Eigen::VectorXd & value(Vals &&... vals) const;

  /** Get the jacbian matrix with respect to x, for the given variable values.
   * See \a value for an explanation of how to specify the values.
   */
  template<typename... Vals>
  const Eigen::MatrixXd & jacobian(const Variable & x, Vals &&... vals) const;

  /** Get the velocity of the function for given values and velocities of its
   * variables.
   * The values and velocities can be given as VectorXd, or as
   * std::initializer_list. There are three possible syntaxes:
   * - velocity(val1, vel1, val2, vel2, ..., valn, veln) where val1, ...,
   *   valn are the values of the n variables of the function, in the order
   *   they are stored in the function (i.e. the order of the vector
   *   returned by variables()) and vel1, ... veln are the corresponding
   *   velocities.
   * - velocity(xi1, vali1, veli1, xi2, vali2, veli2, ...) where one alternates
   *   a reference to one variable and its value and velocity. The number of
   *   variables in this case need not be n (unspecified variables will keep
   *   their previous value), and if a variable is given multiple times,
   *   only the last value will be used
   * - velocity(val, vel) where val and vel are the concatenated values and
   *   velocities for all variables in the order they are stored in the
   *   function.
   */
  template<typename... Vals>
  const Eigen::VectorXd & velocity(Vals &&... vals) const;

  /** Get the normalAcceleration, for the given variable values and
   * velocities.
   * See \a velocity for an explanation of how to specify the values and
   * velocities.
   */
  template<typename... Vals>
  const Eigen::VectorXd & normalAcceleration(Vals &&... vals) const;

  /** Get the time derivative of the jacobian matrix with respect to variable
   * x, for the given variable values and velocities.
   * See \a velocity for an explanation of how to specify the values and
   * velocities.
   */
  template<typename... Vals>
  const Eigen::MatrixXd & JDot(const Variable & x, Vals &&... vals) const;

private:
  static Eigen::VectorXd toVec(std::initializer_list<double> val);

  /** Assign val to the i-th variable. */
  void assign(size_t i, const Eigen::VectorXd & val, bool value) const;
  void assign(Variable & x, const Eigen::VectorXd & val, bool value) const;

  /** Assign all variables with a unique vector. */
  void assign(const Eigen::VectorXd & val) const;
  void assign(const Eigen::VectorXd & val, const Eigen::VectorXd & vel) const;

  /** \internal dispatch the parsing between values only and pairs of
   * variable-value. Initiate the recursion.
   */
  template<typename... Vals>
  void parseValues(const Eigen::VectorXd & v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValues(std::initializer_list<double> v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValues(Variable & x, Vals &&... vals) const;

  /** \internal Recursion over the values.*/
  template<typename... Vals>
  void parseValues_(int i, const Eigen::VectorXd & v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValues_(int i, std::initializer_list<double> v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValues_(Variable & x, const Eigen::VectorXd & v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValues_(Variable & x, std::initializer_list<double> v, Vals &&... vals) const;

  /** \internal Catch wrong number of arguments*/
  template<typename T>
  void parseValues_(T) const;

  /** \internal End of recursion for the parsing of values*/
  void parseValues_(int i, const Eigen::VectorXd & v) const;
  void parseValues_(int i, std::initializer_list<double> v) const;
  void parseValues_(Variable & x, const Eigen::VectorXd & v) const;
  void parseValues_(Variable & x, std::initializer_list<double> v) const;

  /** \internal dispatch the parsing between pairs value-velocity only and
   * triplets variable-value-velocity. Initiate the recursion.
   */
  template<typename... Vals>
  void parseValuesAndVelocities(const Eigen::VectorXd & v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities(std::initializer_list<double> v, Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities(Variable & x, Vals &&... vals) const;

  /** \internal Recursion over the values and velocities.*/
  template<typename... Vals>
  void parseValuesAndVelocities_(int i,
                                 const Eigen::VectorXd & val,
                                 const Eigen::VectorXd & vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(int i,
                                 const Eigen::VectorXd & val,
                                 std::initializer_list<double> vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(int i,
                                 std::initializer_list<double> val,
                                 const Eigen::VectorXd & vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(int i,
                                 std::initializer_list<double> val,
                                 std::initializer_list<double> vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(Variable & x,
                                 const Eigen::VectorXd & val,
                                 const Eigen::VectorXd & vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(Variable & x,
                                 const Eigen::VectorXd & val,
                                 std::initializer_list<double> vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(Variable & x,
                                 std::initializer_list<double> val,
                                 const Eigen::VectorXd & vel,
                                 Vals &&... vals) const;
  template<typename... Vals>
  void parseValuesAndVelocities_(Variable & x,
                                 std::initializer_list<double> val,
                                 std::initializer_list<double> vel,
                                 Vals &&... vals) const;

  /** \internal Catch wrong number of arguments*/
  template<typename T>
  void parseValuesAndVelocities_(T) const;
  template<typename T>
  void parseValuesAndVelocities_(int, T) const;
  template<typename T, typename U>
  void parseValuesAndVelocities_(T, U) const;

  /**  \internal End of recursion.*/
  void parseValuesAndVelocities_(int i, const Eigen::VectorXd & val, const Eigen::VectorXd & vel) const;
  void parseValuesAndVelocities_(int i, const Eigen::VectorXd & val, std::initializer_list<double> vel) const;
  void parseValuesAndVelocities_(int i, std::initializer_list<double> val, const Eigen::VectorXd & vel) const;
  void parseValuesAndVelocities_(int i, std::initializer_list<double> val, std::initializer_list<double> vel) const;
  void parseValuesAndVelocities_(Variable & x, const Eigen::VectorXd & val, const Eigen::VectorXd & vel) const;
  void parseValuesAndVelocities_(Variable & x, const Eigen::VectorXd & val, std::initializer_list<double> vel) const;
  void parseValuesAndVelocities_(Variable & x, std::initializer_list<double> val, const Eigen::VectorXd & vel) const;
  void parseValuesAndVelocities_(Variable & x,
                                 std::initializer_list<double> val,
                                 std::initializer_list<double> vel) const;

  FunctionPtr f_;
  tvm::graph::CallGraph valueGraph_;
  tvm::graph::CallGraph jacobianGraph_;
  tvm::graph::CallGraph velocityGraph_;
  tvm::graph::CallGraph normalAccelerationGraph_;
  tvm::graph::CallGraph JDotGraph_;
  // ensure that derivative will exist.
  std::vector<VariablePtr> dx_;
};

template<typename... Vals>
inline const Eigen::VectorXd & UpdatelessFunction::value(Vals &&... vals) const
{
  using Output = tvm::function::abstract::Function::Output;
  if(f_->isOutputEnabled(Output::Value))
  {
    parseValues(std::forward<Vals>(vals)...);
    valueGraph_.execute();
    return f_->value();
  }
  else
  {
    throw std::runtime_error("Underlying function does not provide a value output.");
  }
}

template<typename... Vals>
inline const Eigen::MatrixXd & UpdatelessFunction::jacobian(const Variable & x, Vals &&... vals) const
{
  using Output = tvm::function::abstract::Function::Output;
  if(f_->isOutputEnabled(Output::Jacobian))
  {
    parseValues(std::forward<Vals>(vals)...);
    jacobianGraph_.execute();
    return f_->jacobian(x);
  }
  else
  {
    throw std::runtime_error("Underlying function does not provide a jacobian output.");
  }
}

template<typename... Vals>
inline const Eigen::VectorXd & UpdatelessFunction::velocity(Vals &&... vals) const
{
  using Output = tvm::function::abstract::Function::Output;
  if(f_->isOutputEnabled(Output::Velocity))
  {
    parseValuesAndVelocities(std::forward<Vals>(vals)...);
    velocityGraph_.execute();
    return f_->velocity();
  }
  else
  {
    throw std::runtime_error("Underlying function does not provide a velocity output.");
  }
}

template<typename... Vals>
inline const Eigen::VectorXd & UpdatelessFunction::normalAcceleration(Vals &&... vals) const
{
  using Output = tvm::function::abstract::Function::Output;
  if(f_->isOutputEnabled(Output::NormalAcceleration))
  {
    parseValuesAndVelocities(std::forward<Vals>(vals)...);
    normalAccelerationGraph_.execute();
    return f_->normalAcceleration();
  }
  else
  {
    throw std::runtime_error("Underlying function does not provide a normalAcceleration output.");
  }
}

template<typename... Vals>
inline const Eigen::MatrixXd & UpdatelessFunction::JDot(const Variable & x, Vals &&... vals) const
{
  using Output = tvm::function::abstract::Function::Output;
  if(f_->isOutputEnabled(Output::JDot))
  {
    parseValuesAndVelocities(std::forward<Vals>(vals)...);
    JDotGraph_.execute();
    return f_->JDot(x);
  }
  else
  {
    throw std::runtime_error("Underlying function does not provide a JDot output.");
  }
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues(const Eigen::VectorXd & v, Vals &&... vals) const
{
  parseValues_(0, v, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues(std::initializer_list<double> v, Vals &&... vals) const
{
  parseValues_(0, v, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues(Variable & x, Vals &&... vals) const
{
  parseValues_(x, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues_(int i, const Eigen::VectorXd & v, Vals &&... vals) const
{
  assign(i, v, true);
  parseValues_(i + 1, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues_(int i, std::initializer_list<double> v, Vals &&... vals) const
{
  parseValues_(i, toVec(v), std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues_(Variable & x, const Eigen::VectorXd & v, Vals &&... vals) const
{
  assign(x, v, true);
  parseValues_(std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValues_(Variable & x, std::initializer_list<double> v, Vals &&... vals) const
{
  parseValues_(x, toVec(v), std::forward<Vals>(vals)...);
}

template<typename T>
inline void UpdatelessFunction::parseValues_(T) const
{
  static_assert(
      ::tvm::internal::always_false<T>::value,
      "Incorrect number of argument. You likely did not observe the alternance between variables and values.");
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities(const Eigen::VectorXd & v, Vals &&... vals) const
{
  parseValuesAndVelocities_(0, v, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities(std::initializer_list<double> v, Vals &&... vals) const
{
  parseValuesAndVelocities_(0, v, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities(Variable & x, Vals &&... vals) const
{
  parseValuesAndVelocities_(x, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(int i,
                                                          const Eigen::VectorXd & val,
                                                          const Eigen::VectorXd & vel,
                                                          Vals &&... vals) const
{
  assign(i, val, true);
  assign(i, vel, false);
  parseValuesAndVelocities_(i + 1, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(int i,
                                                          const Eigen::VectorXd & val,
                                                          std::initializer_list<double> vel,
                                                          Vals &&... vals) const
{
  parseValuesAndVelocities_(i, val, toVec(vel), std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(int i,
                                                          std::initializer_list<double> val,
                                                          const Eigen::VectorXd & vel,
                                                          Vals &&... vals) const
{
  parseValuesAndVelocities_(i, toVec(val), vel, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(int i,
                                                          std::initializer_list<double> val,
                                                          std::initializer_list<double> vel,
                                                          Vals &&... vals) const
{
  parseValuesAndVelocities_(i, toVec(val), toVec(vel), std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(Variable & x,
                                                          const Eigen::VectorXd & val,
                                                          const Eigen::VectorXd & vel,
                                                          Vals &&... vals) const
{
  assign(x, val, true);
  assign(x, vel, false);
  parseValuesAndVelocities_(std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(Variable & x,
                                                          const Eigen::VectorXd & val,
                                                          std::initializer_list<double> vel,
                                                          Vals &&... vals) const
{
  parseValuesAndVelocities_(x, val, toVec(vel), std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(Variable & x,
                                                          std::initializer_list<double> val,
                                                          const Eigen::VectorXd & vel,
                                                          Vals &&... vals) const
{
  parseValuesAndVelocities_(x, toVec(val), vel, std::forward<Vals>(vals)...);
}

template<typename... Vals>
inline void UpdatelessFunction::parseValuesAndVelocities_(Variable & x,
                                                          std::initializer_list<double> val,
                                                          std::initializer_list<double> vel,
                                                          Vals &&... vals) const
{
  parseValuesAndVelocities_(x, toVec(val), toVec(vel), std::forward<Vals>(vals)...);
}

template<typename T>
inline void UpdatelessFunction::parseValuesAndVelocities_(T) const
{
  static_assert(::tvm::internal::always_false<T>::value, "Incorrect number of argument. You likely did not observe the "
                                                         "alternance between variables, values and velocities.");
}

template<typename T>
inline void UpdatelessFunction::parseValuesAndVelocities_(int, T) const
{
  static_assert(::tvm::internal::always_false<T>::value,
                "Incorrect number of argument. You likely forgot a value or velocity.");
}

template<typename T, typename U>
inline void UpdatelessFunction::parseValuesAndVelocities_(T, U) const
{
  static_assert(::tvm::internal::always_false<T>::value, "Incorrect number of argument. You likely did not observe the "
                                                         "alternance between variables, values and velocities.");
}

} // namespace utils

} // namespace tvm
