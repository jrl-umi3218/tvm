#pragma once

#include <memory>

#include <Eigen/Core>

#include <tvm/api.h>
#include <tvm/data/Node.h>

//FIXME add mechanisms for when the function's output is resized

namespace tvm
{
  enum class TDOrder
  {
    Kinematics,
    Dynamics
  };

  class Function;

  class TVM_DLLAPI TaskDynamics : public data::Node<TaskDynamics>
  {
  public:
    SET_OUTPUTS(TaskDynamics, Value)
    SET_UPDATES(TaskDynamics, UpdateValue)

    void setFunction(std::shared_ptr<Function> f);

    const Eigen::VectorXd& value() const;
    TDOrder order() const;

    virtual void UpdateValue() = 0;

  protected:
    TaskDynamics(TDOrder order);
    Function* const function() const;

    Eigen::VectorXd value_;

  private:
    TDOrder order_;
    std::shared_ptr<Function> f_;
  };

  /** Compute -kp*f
    *
    * FIXME have a version with diagonal or sdp gain matrix
    */
  class TVM_DLLAPI ProportionalDynamics : public TaskDynamics
  {
  public:
    ProportionalDynamics(double kp);

    void UpdateValue() override;

  private:
    double kp_;
  };

  /** Compute -kv*dot(f)-kp*f
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

    void UpdateValue() override;

  private:
    double kp_;
    double kv_;
  };
}