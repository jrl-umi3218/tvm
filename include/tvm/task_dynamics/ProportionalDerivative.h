/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

// std::variant is basically unusable in clang 6 + libstdc++ due to a bug
// see, e.g. https://bugs.llvm.org/show_bug.cgi?id=33222
// We use https://github.com/mpark/variant/ instead
#include <mpark/variant.hpp>

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

namespace task_dynamics
{

  /** Compute \f$ \ddot{e}^* = -k_v dot{e} - k_p e\f$ (dynamic order)
   * where \f$ k_p \f$ and \f$ k_v \f$ can be (independently) a scalar, a
   * diagonal matrix (given as a vector) or a matrix.
   */
  class TVM_DLLAPI ProportionalDerivative : public abstract::TaskDynamics
  {
  public:
    using Gain = mpark::variant<double, Eigen::VectorXd, Eigen::MatrixXd>;

    class TVM_DLLAPI Impl: public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Gain& kp, const Gain& kv);
      void updateValue() override;

      ~Impl() override = default;

      /** Get the current gains as a pair (kp, kv). */
      std::pair<const Gain&, const Gain&> gains() const;
      /** Set gains.
        * \param kp Stiffness gain, as a scalar, a vector representing a diagonal matrix, or a matrix.
        * \param kv Damping gain, as a scalar, a vector representing a diagonal matrix, or a matrix.
        *
        * \internal We keep the 9 possible overloads rather than resorting to template to
        *  - have an easily-understandable overview of the different possibilities for the user,
        *  - clarify the types, because the conversion of matrix-like and vector-like expression to
        *    MatrixXd and VectorXd is tricky and does not play well with std::variant. This is also
        *    why we don't use MatrixConstRef and VectorConstRef: conversion to those is ambiguous.
        */
      /**@{*/
      void gains(double kp, double kv);
      void gains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv);
      void gains(const Eigen::MatrixXd& kp, const Eigen::MatrixXd& kv);
      void gains(double kp, const Eigen::VectorXd& kv);
      void gains(double kp, const Eigen::MatrixXd& kv);
      void gains(const Eigen::VectorXd& kp, double kv);
      void gains(const Eigen::VectorXd& kp, const Eigen::MatrixXd& kv);
      void gains(const Eigen::MatrixXd& kp, double kv);
      void gains(const Eigen::MatrixXd& kp, const Eigen::VectorXd& kv);
      /**@}*/

      /** Critically damped version.
        * \param kp Stiffness gain.
        * The damping gain is automatically computed to get a critically damped behavior.
        */
      void gains(double kp);
      /** Critically damped version.
        * \param kp Diagonal of the stiffness gain matrix.
        * The damping gain is automatically computed to get a critically damped behavior.
        */
      void gains(const Eigen::VectorXd& kp);
      /** Critically damped version.
        * \param kp Stiffness gain matrix.
        * The damping gain is automatically computed to get a critically damped behavior.
        * For this version, \p kp is supposed to be symmetric and positive definite.
        *
        * \note The automatic damping computation is relatively heavy, involving a Schur
        * decomposition, matrices multiplications and memory allocation.
        */
      void gains(const Eigen::MatrixXd& kp);

      /** Get the current kp gain*/
      const Gain& kp() const { return kp_; }
      /** Get the current kp gain cast as \p T.
        * \tparam T Type of the gain.
        * \throw Throws if \p T has not the type corresponding to the gain actually used.
        */
      template<typename T>
      const T& kp() const { return mpark::get<T>(kp_); }
      /** Get the current kp gain (non-const version).
        * \warning No check is made if you change the gain. It is your responsability
        * to ensure that its values and its \a size are correct.
        */
      Gain& kp() { return kp_; }
      /** Get the current kp gain cast as \p T (non-const version).
        * \tparam T Type of the gain.
        * \throw Throws if \p T has not the type corresponding to the gain actually used.
        * \warning No check is made if you change the gain. It is your responsability
        * to ensure that its values and its \a size are correct.
        */
      template<typename T>
      T& kp() { return mpark::get<T>(kp_); }

      /** Get the current kv gain*/
      const Gain& kv() const { return kv_; }
      /** Get the current kv gain cast as \p T.
        * \tparam T Type of the gain.
        * \throw Throws if \p T has not the type corresponding to the gain actually used.
        */
      template<typename T>
      const T& kv() const { return mpark::get<T>(kv_); }
      /** Get the current kv gain (non-const version).
        * \warning No check is made if you change the gain. It is your responsability
        * to ensure that its values and its \a size are correct.
        */
      Gain& kv() { return kv_; }
      /** Get the current kv gain cast as \p T (non-const version).
        * \tparam T Type of the gain.
        * \throw Throws if \p T has not the type corresponding to the gain actually used.
        * \warning No check is made if you change the gain. It is your responsability
        * to ensure that its values and its \a size are correct.
        */
      template<typename T>
      T& kv() { return mpark::get<T>(kv_); }

    private:
      void checkGainSize(double k) const;
      void checkGainSize(const Eigen::VectorXd& k) const;
      void checkGainSize(const Eigen::MatrixXd& k) const;
      template<typename T, typename U>
      void gains_(const T& kp, const U& kv);

      Gain kp_; //Stiffness gain
      Gain kv_; //Damping gain
    };

    /** General constructor.
      * \param kp Stiffness gain, as a scalar, a vector representing a diagonal matrix, or a matrix.
      * \param kv Damping gain, as a scalar, a vector representing a diagonal matrix, or a matrix.
      *
      * \internal We keep the 9 possible overloads rather than resorting to template to
      *  - have an easily-understandable overview of the different possibilities for the user,
      *  - clarify the types, because the conversion of matrix-like and vector-like expression to
      *    MatrixXd and VectorXd is tricky and does not play well with std::variant. This is also
      *    why we don't use MatrixConstRef and VectorConstRef: conversion to those is ambiguous.
      */
    /**@{*/
    ProportionalDerivative(double kp, double kv);
    ProportionalDerivative(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv);
    ProportionalDerivative(const Eigen::MatrixXd& kp, const Eigen::MatrixXd& kv);
    ProportionalDerivative(double kp, const Eigen::VectorXd& kv);
    ProportionalDerivative(double kp, const Eigen::MatrixXd& kv);
    ProportionalDerivative(const Eigen::VectorXd& kp, double kv);
    ProportionalDerivative(const Eigen::VectorXd& kp, const Eigen::MatrixXd& kv);
    ProportionalDerivative(const Eigen::MatrixXd& kp, double kv);
    ProportionalDerivative(const Eigen::MatrixXd& kp, const Eigen::VectorXd& kv);
    /**@}*/

    /** Critically damped version.
      * \param kp Stiffness gain.
      * The damping gain is automatically computed to get a critically damped behavior.
      */
    ProportionalDerivative(double kp);
    /** Critically damped version.
      * \param kp Diagonal of the stiffness gain matrix.
      * The damping gain is automatically computed to get a critically damped behavior.
      */
    ProportionalDerivative(const Eigen::VectorXd& kp);
    /** Critically damped version.
      * \param kp Stiffness gain matrix.
      * The damping gain is automatically computed to get a critically damped behavior.
      * For this version, \p kp is supposed to be symmetric and positive definite.
      *
      * \note The automatic damping computation is relatively heavy, involving a Schur
      * decomposition, matrices multiplications and memory allocation.
      */
    ProportionalDerivative(const Eigen::MatrixXd& kp);

    ~ProportionalDerivative() override = default;

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
    Order order_() const override;

  private:
    Gain kp_; //Stiffness gain
    Gain kv_; //Damping gain
  };

  /** Alias for convenience */
  using PD = ProportionalDerivative;

  template<typename T, typename U>
  inline void ProportionalDerivative::Impl::gains_(const T& kp, const U& kv)
  {
    checkGainSize(kp);
    checkGainSize(kv);
    kp_ = kp;
    kv_ = kv;
  }
}  // namespace task_dynamics

}  // namespace tvm
