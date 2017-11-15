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

#include <tvm/api.h>

#include <Eigen/Core>

#include <vector>

namespace tvm
{

namespace robot
{

namespace internal
{

/** Linearized friction cone generator.
 *
 * Compute the vectors that linearize the friction cone using the
 * generatrix.
 */
struct TVM_DLLAPI FrictionCone
{
  /** Default constructor */
  FrictionCone() {}

  /** Compute the friction cone linearization
   *
   * \param frame Friction cone frame. The friction cone is defined along the frame normal axis
   *
   * \param nrGen Number of vectors generating the cone
   *
   * \param mu Coefficient of friction
   *
   * \param dir Cone direction
   *
   */
  FrictionCone(const Eigen::Matrix3d & frame,
               int nrGen, double mu, double direction = 1.0);

  std::vector<Eigen::Vector3d> generators;
};

}

}

}
