/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
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

  /** Compute \f$ \dot{e}^* = -kp e \f$ (Kinematic order)
   *  with \f$ kp \f$ a scalar, a diagonal matrix (given as a vector) or a matrix.
   */
  class TVM_DLLAPI Proportional: public abstract::TaskDynamics
  {
  public:
    using Gain = mpark::variant<double, Eigen::VectorXd, Eigen::MatrixXd>;

    class TVM_DLLAPI Impl: public abstract::TaskDynamicsImpl
    {
    public:
      Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Gain& kp);
      void updateValue() override;

      ~Impl() override = default;

      /** Change the gain to a new scalar. */
      void gain(double kp);
      /** Change the gain to a new diagonal matrix. */
      void gain(const Eigen::VectorXd& kp);
      /** Change the gain to a new matrix. */
      void gain(const  Eigen::MatrixXd& kp);
      /** Get the current gain.*/
      const Gain& gain() const;
      /** Get the current gain cast as \p T.
        * \tparam T Type of the gain.
        * \throw Throws if \p T has not the type corresponding to the gain actually used.
        */
      template<typename T>
      const T& gain() const { return mpark::get<T>(kp_); }
      /** Get the current gain (non-const version).
        * \warning No check is made if you change the gain. It is your responsability
        * to ensure that its values and its \a size are correct.
        */
      Gain& gain();
      /** Get the current gain cast as \p T (non-const version).
        * \tparam T Type of the gain.
        * \throw Throws if \p T has not the type corresponding to the gain actually used.
        * \warning No check is made if you change the gain. It is your responsability
        * to ensure that its values and its \a size are correct.
        */
      template<typename T>
      T& gain() { return mpark::get<T>(kp_); }

    private:
      Gain kp_;
    };

    /** Proportional dynamics with scalar gain. */
    Proportional(double kp);
    /** Proportional dynamics with diagonal matrix gain. */
    Proportional(const Eigen::VectorXd& kp);
    /** Proportional dynamics with matrix gain. */
    Proportional(const Eigen::MatrixXd& kp);

    ~Proportional() override = default;

  protected:
    std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const override;
    Order order_() const override;

    TASK_DYNAMICS_DERIVED_FACTORY(kp_)

  private:
    Gain kp_;
  };

  /** Alias for convenience */
  using P = Proportional;
}  // namespace task_dynamics

}  // namespace tvm
