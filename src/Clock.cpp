#include "Clock.h"
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

void tvm::Clock::reset(double resetTime)
{
  t_ = resetTime;
}

double tvm::Clock::currentTime() const
{
  return t_;
}
