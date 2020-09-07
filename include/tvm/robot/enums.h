/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

namespace tvm
{

namespace robot
{

/** For a given contact, specify the type of constraint that is applied:
 *
 * - Acceleration: constraint the two frames relative-acceleration to remain
 *   the same
 * - Velocity: constraint the two frames relative-velocity to remain the same
 * - Position: constrant the two frames relative-position to remain the same
 *
 * This is done by specifiying different coefficient to the task's dynamics.
 *
 * This only applies to contacts that include a geometric component.
 *
 */
enum class ContactConstraintType
{
  Acceleration,
  Velocity,
  Position
};

/** Specify the type of contact:
 *
 * - Regular: both Force and Geometric
 * - Force: force-only contact (no geometric constraint)
 * - Geometric: geometric-only contact (no additional forces applied by the
 *   robots onto each other)
 *
 */
enum class ContactType
{
  Regular,
  Force,
  Geometric
};

} // namespace robot

} // namespace tvm
