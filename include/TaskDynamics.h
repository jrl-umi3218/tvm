#pragma once

#include <memory>

#include <Eigen/Core>

#include <tvm/api.h>
#include <tvm/data/Node.h>
#include "defs.h"

//FIXME add mechanisms for when the function's output is resized
//FIXME Consider the possibility of having variables in task dynamics?

namespace tvm
{
  enum class TDOrder
  {
    Geometric,
    Kinematics,
    Dynamics
  };

  class TVM_DLLAPI TaskDynamics : public data::Node<TaskDynamics>
  {
  public:
    SET_OUTPUTS(TaskDynamics, Value)
    SET_UPDATES(TaskDynamics, UpdateValue)

    void setFunction(FunctionPtr f);

    const Eigen::VectorXd& value() const;
    TDOrder order() const;

    virtual void updateValue() = 0;

  protected:
    TaskDynamics(TDOrder order);
    Function* const function() const;

    /** Hook for derived class, called at the end of setFunction.*/
    virtual void setFunction_();

    Eigen::VectorXd value_;

  private:
    TDOrder order_;
    FunctionPtr f_;
  };


  /**
    */
  class TVM_DLLAPI NoDynamics : public TaskDynamics
  {
  public:
    NoDynamics(const Eigen::VectorXd& v = Eigen::VectorXd());

    void updateValue() override;

  protected:
    void setFunction_() override;

  private:
    double kp_;
  };

  /** Compute \dot{e}* = -kp*f (Kinematic order)
    *
    * FIXME have a version with diagonal or sdp gain matrix
    */
  class TVM_DLLAPI ProportionalDynamics : public TaskDynamics
  {
  public:
    ProportionalDynamics(double kp);

    void updateValue() override;

  private:
    double kp_;
  };

  /** Compute \ddot{e}* = -kv*dot{f}-kp*f (dynamic order)
  *
  * FIXME have a version with diagonal or sdp gain matrices
  */
  class TVM_DLLAPI ProportionalDerivativeDynamics : public TaskDynamics
  {
  public:
    /** General constructor*/
    ProportionalDerivativeDynamics(double kp, double kv);

    /** Critically damped version*/
    ProportionalDerivativeDynamics(double kp);

    void updateValue() override;

  private:
    double kp_;
    double kv_;
  };


  inline const Eigen::VectorXd& TaskDynamics::value() const
  {
    return value_;
  }

  inline TDOrder TaskDynamics::order() const
  {
    return order_;
  }

  inline Function* const TaskDynamics::function() const
  {
    return f_.get();
  }

  inline void TaskDynamics::setFunction_()
  {
  }
}