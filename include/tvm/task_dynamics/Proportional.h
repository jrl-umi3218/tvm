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
// We use a workaround for std::variant<double, Eigen::VectorXd, Eigen::MatrixXd>
// in this case.
#if defined(__clang__) && __clang_major__ < 7 && defined(__GLIBCXX__)
#include <tvm/task_dynamics/internal/workarounds.h>
#else
#include <variant>
#endif

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

namespace tvm
{

namespace task_dynamics
{

  /** Compute \f$ \dot{e}* = -kp*f \f$ (Kinematic order)
   *  with kp a scalar, a diagonal matrix (given as a vector) or a matrix.
   */
  class TVM_DLLAPI Proportional: public abstract::TaskDynamics
  {
  public:
#if defined(__clang__) && __clang_major__ < 7 && defined(__GLIBCXX__)    
    using Gain = internal::WorkaroundDoubleVectorMatrixVariant;
#else
    using Gain = std::variant<double, Eigen::VectorXd, Eigen::MatrixXd>;
#endif

    class TVM_DLLAPI Impl: public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Gain& kp);
      void updateValue() override;

      /** Change the gain to a new scalar. */
      void gain(double kp);
      /** Change the gain to a new diagonal matrix. */
      void gain(const Eigen::VectorXd& kp);
      /** Change the gain to a new matrix. */
      void gain(const  Eigen::MatrixXd& kp);
      /** Get the current gain.*/
      const Gain& gain() const;

    private:
      Gain kp_;
    };

    /** Proportional dynamics with scalar gain. */
    Proportional(double kp);
    /** Proportional dynamics with diagonal matrix gain. */
    Proportional(const Eigen::VectorXd& kp);
    /** Proportional dynamics with matrix gain. */
    Proportional(const Eigen::MatrixXd& kp);

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
    Order order_() const override;

  private:
    Gain kp_;
  };

  /** Alias for convenience */
  using P = Proportional;
}  // namespace task_dynamics

}  // namespace tvm
