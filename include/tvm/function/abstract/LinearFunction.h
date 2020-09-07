/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

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
