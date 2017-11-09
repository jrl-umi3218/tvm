#pragma once

#include "tvm/data/Outputs.h"

namespace tvm
{
  /* Rationale: we have a unique timer allowed in a ControlProblem (more would
     be possible easily but what for ?). When updating the problem, the following
     order is applied: (i) increment the clock, (ii) call all updateTimeDependency,
     (iii) call the update plan.*/


  class Clock: public data::Outputs
  {
  public:
    SET_OUTPUTS(Clock, CurrentTime)

    Clock(double initTime=0);

    void increment(double dt);
    void reset(double resetTime = 0);
    double currentTime() const;

  private:
    double t_;
  };
}
