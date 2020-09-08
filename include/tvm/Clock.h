/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/defs.h>

#include <tvm/graph/abstract/Outputs.h>

#include <cstdint>

namespace tvm
{

class ControlProblem;

/** Represent a clock for the ControlProblem
 *
 * The current iteration of the problem can be accessed by time-dependant
 * data to trigger computations
 *
 * Outputs:
 *
 *   - Time is moved along as the iterations go on
 *
 */
class TVM_DLLAPI Clock : public graph::abstract::Outputs
{
  friend class ControlProblem;

public:
  SET_OUTPUTS(Clock, Time)

  /** Constructor
   *
   * \param dt Timestep of the ControlProblem
   *
   */
  Clock(double dt);

  /** Returns the number of ticks elapsed since the start of the problem */
  inline uint64_t ticks() const { return ticks_; }

  /** Returns the timestep of the problem */
  inline double dt() const { return dt_; }

  /** Advance the clock by one tick */
  void advance();

protected:
  Clock(const Clock &) = default;
  Clock & operator=(const Clock &) = default;
  Clock(Clock &&) = default;
  Clock & operator=(Clock &&) = default;

private:
  const double dt_;
  uint64_t ticks_ = 0;
};
} // namespace tvm
