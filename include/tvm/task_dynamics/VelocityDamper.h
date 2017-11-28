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
    /** A first or second order dynamic task implementing the so-called velocity
      * damper of Faverjon and Tournassoud.
      * For a lower bound tasks, we have:
      *  - first order: dot{e}* = -xsi * (e-ds)/(di-ds)
      *  - second order: ddot{e}* = -xsi/dt * (e-ds)/(di-ds) -dot{e}/dt
      * For upper bound tasks e <= 0 this is adapted to get the same behavior as
      * -e >= 0.
      */
    class TVM_DLLAPI VelocityDamper : public abstract::TaskDynamics
    {
    public:
      class TVM_DLLAPI Impl : public abstract::TaskDynamicsImpl
      {
      public:
        //First order dynamics
        Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, double di, double ds, double xsi, double big);
        //Second order dynamics
        Impl(FunctionPtr, constraint::Type t, const Eigen::VectorXd& rhs, double dt, double di, double ds, double xsi, double big);
        
        void updateValue() override;

      private:
        void compute_ab();

        double dt_;
        double xsi_;
        double ds_;
        double di_;
        double a_;
        double b_;
        double big_;

        Eigen::VectorXd d_;
      };

      VelocityDamper(double di, double ds, double xsi, double big = constant::big_number);
      VelocityDamper(double dt, double di, double ds, double xsi, double big = constant::big_number);

    protected:
      std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;

    private:
      double dt_;
      double xsi_;
      double ds_;
      double di_;
      double big_;
    };

  }  // namespace task_dynamics

}  // namespace tvm
