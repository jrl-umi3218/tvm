#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

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

}

}
