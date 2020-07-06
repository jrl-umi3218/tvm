/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#include <Eigen/Eigenvalues>

#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/function/abstract/Function.h>

namespace tvm
{

namespace task_dynamics
{

  ProportionalDerivative::ProportionalDerivative(double kp, double kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::MatrixXd& kp, const Eigen::MatrixXd& kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(double kp, const Eigen::VectorXd& kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(double kp, const Eigen::MatrixXd& kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::VectorXd& kp, double kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::VectorXd& kp, const Eigen::MatrixXd& kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::MatrixXd& kp, double kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::MatrixXd& kp, const Eigen::VectorXd& kv)
    : kp_(kp)
    , kv_(kv)
  {
  }

  ProportionalDerivative::ProportionalDerivative(double kp)
    : ProportionalDerivative(kp, 2 * std::sqrt(kp))
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::VectorXd& kp)
    : kp_(kp)
    , kv_(mpark::in_place_index<1>, 2*kp.cwiseSqrt())
  {
  }

  ProportionalDerivative::ProportionalDerivative(const Eigen::MatrixXd& kp)
    : kp_(kp)
    , kv_(Eigen::MatrixXd(kp.rows(), kp.cols()))
  {
    Eigen::RealSchur<Eigen::MatrixXd> dec(kp);
    assert(dec.matrixT().isDiagonal(1e-8) && "kp is not symmetric.");
    assert((dec.matrixT().diagonal().array() >= 0).all() && "kp is undefinite.");
    kv_.emplace<2>(2 * dec.matrixU() * dec.matrixT().diagonal().asDiagonal() * dec.matrixU().transpose());
  }

  std::unique_ptr<abstract::TaskDynamicsImpl> ProportionalDerivative::impl_(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs) const
  {
    return std::make_unique<Impl>(f, t, rhs, kp_, kv_);
  }

  Order ProportionalDerivative::order_() const
  {
    return Order::Two;
  }

  ProportionalDerivative::Impl::Impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd& rhs, const Gain& kp, const Gain& kv)
    : TaskDynamicsImpl(Order::Two, f, t, rhs)
    , kp_(kp)
    , kv_(kv)
  {
    assert((kp.index() == 0                                           // Scalar gain
        || (kp.index() == 1 && mpark::get<Eigen::VectorXd>(kp).size() == f->size())   // Diagonal gain
        || (kp.index() == 2 && mpark::get<Eigen::MatrixXd>(kp).cols() == f->size()    // Matrix gain
                            && mpark::get<Eigen::MatrixXd>(kp).rows() == f->size()))  
      && "Gain kp and function have incompatible sizes");

    assert((kv.index() == 0                                           // Scalar gain
        || (kv.index() == 1 && mpark::get<Eigen::VectorXd>(kv).size() == f->size())   // Diagonal gain
        || (kv.index() == 2 && mpark::get<Eigen::MatrixXd>(kv).cols() == f->size()    // Matrix gain
                            && mpark::get<Eigen::MatrixXd>(kv).rows() == f->size()))  
      && "Gain kv and function have incompatible sizes");
  }

  void ProportionalDerivative::Impl::updateValue()
  {
    switch (kv_.index())
    {
    case 0: value_ = -mpark::get<double>(kv_) * function().velocity(); break;
    case 1: value_.noalias() = -(mpark::get<Eigen::VectorXd>(kv_).asDiagonal() * function().velocity()); break;
    case 2: value_.noalias() = -mpark::get<Eigen::MatrixXd>(kv_) * function().velocity(); break;
    default: assert(false);
    }
    switch (kp_.index())
    {
    case 0: value_ -= mpark::get<double>(kp_) * (function().value() - rhs()); break;
    case 1: value_.noalias() -= mpark::get<Eigen::VectorXd>(kp_).asDiagonal() * (function().value() - rhs()); break;
    case 2: value_.noalias() -= mpark::get<Eigen::MatrixXd>(kp_) * (function().value() - rhs()); break;
    default: assert(false);
    }
  }

  std::pair<const PD::Gain&, const PD::Gain&> ProportionalDerivative::Impl::gains() const
  {
    return {kp_, kv_};
  }

  void ProportionalDerivative::Impl::gains(double kp, double kv)
  {
    gains_(kp, kv);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::MatrixXd& kp, const Eigen::MatrixXd& kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(double kp, const Eigen::VectorXd& kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(double kp, const Eigen::MatrixXd& kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::VectorXd& kp, double kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::VectorXd& kp, const Eigen::MatrixXd& kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::MatrixXd& kp, double kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::MatrixXd& kp, const Eigen::VectorXd& kv)
  {
    gains_(kp,kv);
  }

  void ProportionalDerivative::Impl::gains(double kp)
  {
    kp_ = kp;
    kv_ = 2 * std::sqrt(kp);
  }

  void ProportionalDerivative::Impl::gains(const Eigen::VectorXd& kp)
  {
    checkGainSize(kp);
    kp_ = kp;
    kv_.emplace<1>(2 * kp.cwiseSqrt());
  }

  void ProportionalDerivative::Impl::gains(const Eigen::MatrixXd& kp)
  {
    checkGainSize(kp);
    Eigen::RealSchur<Eigen::MatrixXd> dec(kp);
    assert(dec.matrixT().isDiagonal(1e-8) && "kp is not symmetric.");
    assert((dec.matrixT().diagonal().array() >= 0).all() && "kp is undefinite.");
    kv_.emplace<2>(2*dec.matrixU() * dec.matrixT().diagonal().asDiagonal() * dec.matrixU().transpose());
  }

  void ProportionalDerivative::Impl::checkGainSize(double k) const
  {
    //do nothing
  }

  void ProportionalDerivative::Impl::checkGainSize(const Eigen::VectorXd& k) const
  {
    if (k.size() != function().size())
    {
      throw std::runtime_error("[task_dynamics::ProportionalDerivative::Impl::gains] Gain and function have incompatible sizes.");
    }
  }

  void ProportionalDerivative::Impl::checkGainSize(const Eigen::MatrixXd& k) const
  {
    if (k.rows() != function().size() || k.cols() != function().size())
    {
      throw std::runtime_error("[task_dynamics::ProportionalDerivative::Impl::gains] Gain and function have incompatible sizes.");
    }
  }

}  // namespace task_dynamics

}  // namespace tvm
