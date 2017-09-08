#pragma once

#include "LinearConstraint.h"
#include "Task.h"

namespace tvm
{
  /** Given a task (e, op, e*), this class derives the constraint 
    * d^k e/dt^k op  e*(e,de/dt,...d^{k-1}/dt^{k-1} [,g]), where e is an error 
    * function, op is ==, >= or <= and e* is a desired error dynamics, k is 
    * specified by e* and (optional) g is any other quantities.
    *
    * FIXME Consider the case where the TaskDynamics has its own variables?
    *
    */
  class TVM_DLLAPI LinearizedTaskConstraint : public LinearConstraint
  {
  public:
    SET_UPDATES(LinearizedTaskConstraint, UpdateRHS)

    LinearizedTaskConstraint(const Task& task);
    LinearizedTaskConstraint(const ProtoTask& pt, std::shared_ptr<TaskDynamics> td);

    void updateLKin();
    void updateLDyn();
    void updateUKin();
    void updateUDyn();
    void updateEKin();
    void updateEDyn();

    const MatrixWithProperties& jacobian(const Variable& x) const override;

  private:
    FunctionPtr f_;
    std::shared_ptr<TaskDynamics> td_;
  };
}