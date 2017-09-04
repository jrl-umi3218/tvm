#pragma once

#include <memory>

namespace tvm
{
  /* Rationale: we have a unique timer allowed in a ControlProblem (more would 
     be possible easily but what for ?). When updating the problem, the following
     order is applied: (i) increment the clock, (ii) call all updateTimeDependency,
     (iii) call the update plan.*/


  class Clock
  {
  public:
    Clock(double initTime=0);

    void increment(double dt);
    double currentTime() const;

  private:
    double t_;
  };


  class ExplicitlyTimeDependent
  {
  public:
    ExplicitlyTimeDependent(std::shared_ptr<Clock> clock);

    void updateTimeDependency();

  protected:
    virtual void updateTimeDependency_() = 0;

    std::shared_ptr<Clock> clock_;

  };
}