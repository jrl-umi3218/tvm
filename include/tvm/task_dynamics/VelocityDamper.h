/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/task_dynamics/abstract/TaskDynamics.h>

#include <optional>

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
  /** A structure grouping the parameters of a velocity damper.
   * \sa VelocityDamper.
   */
  class TVM_DLLAPI Config
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
    Config(double di, double ds, double xsi, double xsiOff = 0);

    double di_;
    double ds_;
    double xsi_;
    double xsiOff_;
  };

  /** A structure grouping the parameters for an anisotropic velocity damper.
   * \sa VelocityDamper
   */
  class TVM_DLLAPI AnisotropicConfig
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
     *
     * All parameters must be of the same size
     */
    AnisotropicConfig(const VectorConstRef & di,
                      const VectorConstRef & ds,
                      const VectorConstRef & xsi,
                      const std::optional<VectorConstRef> & xsiOff = std::nullopt);

    /** Construct from a non-anisotropic configuration
     *
     * \param config Configuration to use
     */
    AnisotropicConfig(const Config & config);

    Eigen::VectorXd di_;
    Eigen::VectorXd ds_;
    Eigen::VectorXd xsi_;
    Eigen::VectorXd xsiOff_;
  };

  class TVM_DLLAPI Impl : public abstract::TaskDynamicsImpl
  {
  public:
    // First order dynamics
    Impl(FunctionPtr f,
         constraint::Type t,
         const Eigen::VectorXd & rhs,
         bool autoXsi,
         const Eigen::VectorXd & di,
         const Eigen::VectorXd & ds,
         const Eigen::VectorXd & xsi,
         double big);
    // Second order dynamics
    Impl(FunctionPtr f,
         constraint::Type t,
         const Eigen::VectorXd & rhs,
         double dt,
         bool autoXsi,
         const Eigen::VectorXd & di,
         const Eigen::VectorXd & ds,
         const Eigen::VectorXd & xsi,
         double big);

    void updateValue() override;

    ~Impl() override = default;

  private:
    /** Partial update computation. \p s is a sign factor to apply on the velocity.*/
    void updateValue_(double s);

    double dt_;
    Eigen::VectorXd ds_;
    Eigen::VectorXd di_;
    Eigen::VectorXd xsiOff_;
    Eigen::VectorXd a_; // -1 / (di - ds)
    double big_;

    bool autoXsi_;
    Eigen::VectorXd d_;
    Eigen::VectorXd axsi_; // a * xsi = -xsi / (di - ds)
    std::vector<bool> active_;
  };

  /** \brief Velocity damper for first order dynamics.
   *
   * \param config configuration of the damper. \sa VelocityDamperConfig
   *
   * \param big value used as infinity.
   *
   * \attention When \p autoXsi is \p true, the value update will declare a
   * dependency to the \p Velocity of the error function. It is the user
   * responsibility to ensure the velocity is correctly provided (in
   * particular, this might require the value of the variables first
   * derivatives to be set correctly).
   */
  VelocityDamper(const Config & config, double big = constant::big_number);

  /** \brief Velocity damper for first order dynamics.
   *
   * \param config configuration of the damper. \sa VelocityDamperAnisotropicConfig
   *
   * \param big value used as infinity.
   *
   * \attention Previous remark on \p autoXsi apply. Furthermore, the
   * dimensions of \p config must match the function for which this task
   * dynamic will be used.
   */
  VelocityDamper(const AnisotropicConfig & config, double big = constant::big_number);

  /** \brief Velocity damper for second order dynamics.
   *
   * \param dt integration time step to integrate acceleration to velocity
   * (should be the control time step). Must be strictly positive.
   *
   * \param config configuration of the damper. \sa VelocityDamperConfig
   *
   * \param big value used as infinity.
   */
  VelocityDamper(double dt, const Config & config, double big = constant::big_number);

  /** \brief Velocity damper for second order dynamics.
   *
   * \param dt integration time step to integrate acceleration to velocity
   * (should be the control time step). Must be strictly positive.
   *
   * \param config configuration of the damper. \sa VelocityDamperAnisotropicConfig
   *
   * \param big value used as infinity.
   */
  VelocityDamper(double dt, const AnisotropicConfig & config, double big = constant::big_number);

  ~VelocityDamper() override = default;

protected:
  std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f,
                                                    constraint::Type t,
                                                    const Eigen::VectorXd & rhs) const override;
  Order order_() const override;

  template<typename Derived, typename... Args>
  std::unique_ptr<abstract::TaskDynamicsImpl> impl_(FunctionPtr f,
                                                    constraint::Type t,
                                                    const Eigen::VectorXd & rhs,
                                                    Args &&... args) const
  {
    if(dt_ > 0)
    {
      return std::make_unique<Derived>(f, t, rhs, std::forward<Args>(args)..., dt_, autoXsi_, di_, ds_, xsi_, big_);
    }
    return std::make_unique<Derived>(f, t, rhs, std::forward<Args>(args)..., autoXsi_, di_, ds_, xsi_, big_);
  }

private:
  double dt_;
  Eigen::VectorXd xsi_;
  Eigen::VectorXd ds_;
  Eigen::VectorXd di_;
  double big_;
  bool autoXsi_;
};

} // namespace task_dynamics

} // namespace tvm
