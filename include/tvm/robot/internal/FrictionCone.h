/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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
 * generators.
 */
class TVM_DLLAPI FrictionCone
{
public:
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
  FrictionCone(const Eigen::Matrix3d & frame, unsigned int nrGen, double mu, double direction = 1.0);

  std::vector<Eigen::Vector3d> generators;
};

} // namespace internal

} // namespace robot

} // namespace tvm
