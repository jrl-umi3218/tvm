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

#include <Eigen/Core>

#include <initializer_list>
#include <memory>

namespace tvm
{

namespace function
{

namespace abstract
{

/** Base class for linear functions.
 *
 * \dot
 * digraph "update graph" {
 *   rankdir="LR";
 *   {
 *     rank=same; node [shape=circle];
 *     x_i; dx_i;
 *   }
 *   {
 *     rank = same; node [shape=hexagon];
 *     Value; Jacobian; Velocity;
 *     NormalAcceleration; JDot; B
 *   }
 *   {
 *     rank = same;
 *     uValue [label=Value];
 *     uVelocity [label=Velocity];
 *   }
 *   {
 *     rank = same; node [style=invis, label=""];
 *     outValue; outJacobian; outVelocity;
 *     outNormalAcceleration; outJDot; outB
 *   }
 *   x_i -> uValue [label="value()"];
 *   dx_i -> uVelocity [label="value()"];
 *   Jacobian -> uValue;
 *   B -> uValue;
 *   Jacobian -> uVelocity;
 *   uValue -> Value;
 *   uVelocity -> Velocity;
 *   B -> outB [label="b()"];
 *   Value -> outValue [label="value()"];
 *   Jacobian -> outJacobian [label="jacobian(x_i)"];
 *   Velocity -> outVelocity [label="velocity()"];
 *   NormalAcceleration -> outNormalAcceleration [label="normalAcceleration()"];
 *   JDot -> outJDot [label="JDot(x_i)"];
 * }
 * \enddot
 */
class TVM_DLLAPI LinearFunction : public Function
{
public:
  SET_OUTPUTS(LinearFunction, B)
  SET_UPDATES(LinearFunction, Value, Velocity)

  void updateValue();
  void updateVelocity();
  void resizeCache() override;
  const internal::VectorWithProperties & b() const { return b_; }

protected:
  LinearFunction(int m);
  virtual void updateValue_();
  virtual void updateVelocity_();
  void setDerivativesToZero();

  internal::VectorWithProperties b_;
};

} // namespace abstract

} // namespace function

} // namespace tvm
