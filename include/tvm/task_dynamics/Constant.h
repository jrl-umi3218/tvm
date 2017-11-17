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

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

  namespace task_dynamics
  {
    class TVM_DLLAPI Constant : public abstract::TaskDynamics
    {
    public:
      class TVM_DLLAPI Impl: public abstract::TaskDynamicsImpl
      {
      public:
        Impl(FunctionPtr, constraint::Type t, const Eigen::VectorXd& rhs, const Eigen::VectorXd& v);
        void updateValue() override;
      };

      Constant(const Eigen::VectorXd& v = Eigen::VectorXd());

    protected:
      std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;

    private:
      Eigen::VectorXd v_;
    };

  }  // namespace task_dynamics

}  // namespace tvm
