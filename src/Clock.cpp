#include <tvm/Clock.h>

#include <cassert>

namespace tvm
{

Clock::Clock(double dt)
: dt_(dt)
{
  assert(dt > 0);
}

void Clock::tick()
{
  ticks_++;
}

} // namespace tvm
