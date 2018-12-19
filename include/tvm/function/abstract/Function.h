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

#include <tvm/internal/FirstOrderProvider.h>
#include <tvm/utils/internal/map.h>

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
    utils::internal::map<Variable const *, Eigen::MatrixXd> JDot_;

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
