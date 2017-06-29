#include "Time.h"
#include <assert.h>

taskvm::Clock::Clock(double initTime)
  : t_(initTime)
{
}

void taskvm::Clock::increment(double dt)
{
  assert(dt > 0);
  t_ += dt;
}

double taskvm::Clock::currentTime() const
{
  return t_;
}

taskvm::ExplicitlyTimeDependent::ExplicitlyTimeDependent(std::shared_ptr<Clock> clock)
  : clock_(clock)
{
}

void taskvm::ExplicitlyTimeDependent::updateTimeDependency()
{
  updateTimeDependency_();
}
