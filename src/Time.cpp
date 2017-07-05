#include "Time.h"
#include <assert.h>

tvm::Clock::Clock(double initTime)
  : t_(initTime)
{
}

void tvm::Clock::increment(double dt)
{
  assert(dt > 0);
  t_ += dt;
}

double tvm::Clock::currentTime() const
{
  return t_;
}

tvm::ExplicitlyTimeDependent::ExplicitlyTimeDependent(std::shared_ptr<Clock> clock)
  : clock_(clock)
{
}

void tvm::ExplicitlyTimeDependent::updateTimeDependency()
{
  updateTimeDependency_();
}
