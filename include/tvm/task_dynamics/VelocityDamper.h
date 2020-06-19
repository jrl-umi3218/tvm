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

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

  namespace task_dynamics
  {
    /** A structure grouping the parameters of a velocity damper.
      * \sa VelocityDamper.
      */
    class TVM_DLLAPI VelocityDamperConfig
    {
    public:
      /**
        * \param di interaction distance \f$d_i\f$. We need \f$ d_i > d_s \f$.
        * \param ds safety distance \f$d_s\f$.
        * \param xsi damping parameter \f$\xi \f$. If xsi = 0, the value will
        * be computed automatically, otherwise, we need \f$\xi > 0\f$.
        * In automatic mode, the value is recomputed each time the error value
        * is at a distance to its bound lower or equal to \p \di with the 
        * formula \f$ \xi = -\dfrac{d_i - d_s}{d^k - d_s} \dot{d}^k + 
        * \xi_{\mathrm{off}} \f$.
        * \param xsiOff offset \f$ \xi_{\mathrm{off}} \f$ used in the automatic
        * computation of \f$\xi\f$. Used only in the case xsi=0.
        */
      VelocityDamperConfig(double di, double ds, double xsi, double xsiOff=0);

      double di_;
      double ds_;
      double xsi_;
      double xsiOff_;
    };

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

        ~Impl() override = default;

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
        * \param config configuration of the damper. \sa VelocityDamperConfig
        * \param big value used as infinity.
        * \attention When \p autoXsi is \p true, the value update will declare a
        * dependency to the \p Velocity of the error function. It is the user
        * responsibility to ensure the velocity is correctly provided (in
        * particular, this might require the value of the variables first
        * derivatives to be set correctly).
        */
      VelocityDamper(const VelocityDamperConfig& config, double big = constant::big_number);
      
      /** \brief Velocity damper for second order dynamics.
        *
        * \param dt integration time step to integrate acceleration to velocity
        * (should be the control time step). Must be non-negative.
        * \param config configuration of the damper. \sa VelocityDamperConfig
        * \param big value used as infinity.
        */
      VelocityDamper(double dt, const VelocityDamperConfig& config, double big = constant::big_number);

      ~VelocityDamper() override = default;

    protected:
      std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
      Order order_() const override;

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
