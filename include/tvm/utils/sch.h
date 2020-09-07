/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

/** @name SCH utilities
 *
 * These functions make it easier to work with sch-core
 *
 * @{
 */
#include <tvm/api.h>

#include <sch/CD/CD_Pair.h>
#include <sch/S_Polyhedron/S_Polyhedron.h>

#include <SpaceVecAlg/SpaceVecAlg>

#include <memory>

namespace tvm
{

namespace utils
{

/** Set \p obj pose to \p t */
TVM_DLLAPI void transform(sch::S_Object & obj, const sva::PTransformd & t);

/** Loads an sch::S_Polyhedron object using \p filename */
TVM_DLLAPI std::unique_ptr<sch::S_Polyhedron> Polyhedron(const std::string & filename);

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
TVM_DLLAPI double distance(sch::CD_Pair & pair, Eigen::Vector3d & p1, Eigen::Vector3d & p2);

} // namespace utils

} // namespace tvm

/** @} */
/* End of SCH utilities */
