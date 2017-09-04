#pragma once

#include <map>

#include <Eigen/Core>

#include "FirstOrderProvider.h"

namespace tvm
{
  /* Notes: we define the classical outputs for a function*/

  class Function : public internal::FirstOrderProvider
  {
  public:
    SET_OUTPUTS(Function, Velocity, NormalAcceleration, JDot);

    const Eigen::VectorXd& velocity() const;
    const Eigen::VectorXd& normalAcceleration() const;
    const Eigen::MatrixXd& JDot(const Variable& x) const;

    /** Note: by default, these methods return the cached value.
      * However, they are virtual in case the user might want to bypass the cache.
      * This would be typically the case if he/she wants to directly return the
      * output of another method, e.g. return the jacobian of an other Function.
      * 
      * Question: should they be made protected or stay public
      */
    virtual const Eigen::VectorXd& velocityNoCheck() const;
    virtual const Eigen::VectorXd& normalAccelerationNoCheck() const;
    virtual const Eigen::MatrixXd& JDotNoCheck(const Variable& x) const;

  protected:
    Function();

    /** Resize all cache members corresponding to active output*/
    void resizeCache() override;

    virtual void addVariable_(VariablePtr v) override;
    virtual void removeVariable_(VariablePtr v) override;

  private:
    // cache
    Eigen::VectorXd velocity_;
    Eigen::VectorXd normalAcceleration_;
    std::map<Variable const *, Eigen::MatrixXd> JDot_;
  };
}