#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */


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
    *     uValue [label=Value];
    *     uVelocity [label=Velocity];
    *   }
    *   {
    *     rank = same; node [shape=hexagon];
    *     Value; Jacobian; Velocity;
    *     NormalAcceleration; JDot;
    *   }
    *   {
    *     rank = same; node [style=invis, label=""];
    *     outValue; outJacobian; outVelocity;
    *     outNormalAcceleration; outJDot;
    *   }
    *   x_i -> uValue [label="value()"];
    *   dx_i -> uVelocity [label="value()"];
    *   uValue -> Value;
    *   uVelocity -> Velocity;
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
    SET_UPDATES(LinearFunction, Value, Velocity)

    void updateValue();
    void updateVelocity();
    void resizeCache() override;
    const Eigen::VectorXd& b() const;

  protected:
    LinearFunction(int m);
    virtual void updateValue_();
    virtual void updateVelocity_();
    void setDerivativesToZero();

    Eigen::VectorXd b_;
  };

  inline const Eigen::VectorXd& LinearFunction::b() const
  {
    return b_;
  }


}  // namespace abstract

}  // namespace function

}  // namespace tvm
