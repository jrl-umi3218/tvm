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

#include <tvm/internal/FirstOrderProvider.h>

#include <Eigen/Core>

#include <map>

namespace tvm
{

namespace function
{

namespace abstract
{

  /** Base class defining the classical outputs for a function
    *
    * \dot
    * digraph "update graph" {
    *   rankdir="LR";
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
    *   Value -> outValue [label="value()"];
    *   Jacobian -> outJacobian [label="jacobian(x_i)"];
    *   Velocity -> outVelocity [label="velocity()"];
    *   NormalAcceleration -> outNormalAcceleration [label="normalAcceleration()"];
    *   JDot -> outJDot [label="JDot(x_i)"];
    * }
    * \enddot
    */
  class TVM_DLLAPI Function : public tvm::internal::FirstOrderProvider
  {
  public:
    SET_OUTPUTS(Function, Velocity, NormalAcceleration, JDot)

    /** Note: by default, these methods return the cached value.
      * However, they are virtual in case the user might want to bypass the cache.
      * This would be typically the case if he/she wants to directly return the
      * output of another method, e.g. return the jacobian of an other Function.
      */
    virtual const Eigen::VectorXd& velocity() const;
    virtual const Eigen::VectorXd& normalAcceleration() const;
    virtual const Eigen::MatrixXd& JDot(const Variable& x) const;

  protected:
    /** Constructor
      * /param m the output size of the function, i.e. the size of the value (or
      * equivalently the row size of the jacobians).
      */
    Function(int m=0);

    /** Resize all cache members corresponding to active output*/
    void resizeCache() override;
    void resizeVelocityCache();
    void resizeNormalAccelerationCache();
    void resizeJDotCache();

    void addVariable_(VariablePtr v) override;
    void removeVariable_(VariablePtr v) override;

    // cache
    Eigen::VectorXd velocity_;
    Eigen::VectorXd normalAcceleration_;
    std::map<Variable const *, Eigen::MatrixXd> JDot_;

  private:
    //we retain the variables' derivatives shared_ptr to ensure the reference is never lost
    std::vector<VariablePtr> variablesDot_;
  };


  inline const Eigen::VectorXd& Function::velocity() const
  {
    return velocity_;
  }

  inline const Eigen::VectorXd& Function::normalAcceleration() const
  {
    return normalAcceleration_;
  }

  inline const Eigen::MatrixXd& Function::JDot(const Variable& x) const
  {
    return JDot_.at(&x);
  }

}  // namespace abstract

}  // namespace function

}  // namespace tvm
