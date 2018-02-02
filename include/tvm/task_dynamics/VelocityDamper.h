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
      * For a lower bound tasks e>=0, we have, for e<=di:
      *  - first order: dot{e}_i* = -xsi * (e_i-ds)/(di-ds)
      *  - second order: ddot{e}_i* = -xsi/dt * (e_i-ds)/(di-ds) -dot{e}_i/dt
      *
      * and for e>di:
      *   - first order: dot{e}* = big
      *   - second order: ddot{e}* = big
      *
      * For upper bound tasks e <= 0 this is adapted to get the same behavior as
      * -e >= 0.
      *
      * xsi can be computed automatically (see \p autoXsi parameter).
      *
      * \attention For xsi to be computed automatically in the first order case,
      * the value of dot{e} is used.
      *
      * \attention To know when to compute xsi in automatic mode, the class keeps
      * data on the last iterations. Consequently if the velocity damper is
      * applied to a function \p f with variable \p x, its output value will be
      * the same if \p x keeps the same value \p x0 and velocity \p v0 between
      * two consecutive updates, but the same entries \p x0 and \p v0 \a may give
      * different values for unrelated updates, because of history differences.
      * \n This remark only applies when the automatic computation of xsi is
      * required.
      */
    class TVM_DLLAPI VelocityDamper : public abstract::TaskDynamics
    {
    public:
      class TVM_DLLAPI Impl : public abstract::TaskDynamicsImpl
      {
      public:
        //First order dynamics
        Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, bool autoXsi, double di, double ds, double xsi, double big);
        //Second order dynamics
        Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, double dt, bool autoXsi, double di, double ds, double xsi, double big);
        
        void updateValue() override;

      private:
        /** Partial update computation. \p s is a sign factor to apply on the velocity.*/
        void updateValue_(double s);

        double dt_;
        double ds_;
        double di_;
        double xsiOff_;
        double a_;                  // -1 / (di - ds)
        double big_;

        bool autoXsi_;
        Eigen::VectorXd d_;
        Eigen::VectorXd axsi_;      //a * xsi = -xsi / (di - ds)
        std::vector<bool> active_;
      };

      /** \bried Velocity damper for first order dynamics.
        * For parameters, see VelocityDamper::VelocityDamper(double, bool, double, double, double, double)
        * \attention When \p autoXsi is \p true, the value update will declare a
        * dependency to the \p Velocity of the error function. It is the user
        * responsibility to ensure the velocity is correctly provided (in
        * particular, this might require the value of the variables first
        * derivatives to be set correctly).
        */
      VelocityDamper(bool autoXsi, double di, double ds, double xsi, double big = constant::big_number);
      
      /** \brief Velocity damper for second order dynamics.
        *
        * \param dt integration time step to integrate acceleration to velocity
        * (should be the control time step). Must be non-negative.
        * \param autoXsi when \p false, the computation of the bound uses the
        * value \p xsi given by the user. When \p true the value is recomputed
        * each time the error value is at a distance to its bound lower or equal
        * to \p \di with the formula \f$ \xi = -\dfrac{d_i - d_s}{d^k - d_s} 
        * \dot{d}^k + \xi_{\mathrm{off}} \f$. In this case, the user-given
        * parameter \p xsi corresponds to \f$ \xi_{\mathrm{off}} \f$.
        * \param di interaction distance \f$d_i\f$. We need \f$ d_i > d_s \f$.
        * \param ds safety distance \f$d_s\f$.
        * \param xsi damping parameter \f$\xi \f$. We need \f$\xi > 0\f$ if \p 
        * autoXsi is \p false and \f$\xi \geq 0\f$ otherwise.
        * \param big value used as infinity.
        *
        */
      VelocityDamper(double dt, bool autoXsi, double di, double ds, double xsi, double big = constant::big_number);

    protected:
      std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;

    private:
      double dt_;
      double xsi_;
      double ds_;
      double di_;
      double big_;
      bool autoXsi_;
    };

  }  // namespace task_dynamics

}  // namespace tvm
