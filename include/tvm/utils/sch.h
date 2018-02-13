#pragma once

/* Copyright 2018 CNRS-UM LIRMM, CNRS-AIST JRL
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

/** @name SCH utilities
 *
 * These functions make it easier to work with sch-core
 *
 * @{
 */
#include <tvm/api.h>

#include <sch/S_Polyhedron/S_Polyhedron.h>
#include <sch/CD/CD_Pair.h>

#include <SpaceVecAlg/SpaceVecAlg>

#include <memory>

namespace tvm
{

namespace utils
{

  /** Set \p obj pose to \p t */
  TVM_DLLAPI void transform(sch::S_Object& obj, const sva::PTransformd& t);

  /** Loads an sch::S_Polyhedron object using \p filename */
  TVM_DLLAPI std::unique_ptr<sch::S_Polyhedron> Polyhedron(const std::string& filename);

  /** Compute distance between two objects
   *
   * \param pair pair of SCH objects
   *
   * \param p1 will be the closest point in the first object
   *
   * \param p2 will be the closes point in the second object
   *
   * \returns The squared distance between the two objects, returns a negative
   * distance in case of inter-penetration
   *
   */
  TVM_DLLAPI double distance(sch::CD_Pair& pair, Eigen::Vector3d& p1, Eigen::Vector3d& p2);

} // utils

} // namespace tvm

/** @} */
/* End of SCH utilities */
