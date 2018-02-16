#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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
#include <tvm/defs.h>
#include <tvm/graph/abstract/Node.h>

namespace tvm
{

namespace geometry
{

  /** Represents a 2D plane in 3D space
   *
   * A static plane is represented by the usual (normal, offset)
   * representation.
   *
   * A moving plane is represented by the (normal, point) couple and their
   * derivatives.
   *
   * This object does not update any of these quantities (exception: when point
   * is changed, offset is changed accordingly), this reponsability belongs to
   * the Plane provider. In other words, this is *not* a plane integrator.
   *
   * Outputs:
   * - Position: position of the plane, represents (normal, offset, point)
   *   quantities
   * - Velocity: first derivative of (normal, point)
   * - Acceleration: second derivative of (normal, point)
   *
   */

  class TVM_DLLAPI Plane : public graph::abstract::Node<Plane>
  {
  public:
    SET_OUTPUTS(Plane, Position, Velocity, Acceleration)

    /** Constructor for a static plane
     *
     * \param normal Normal of the plane
     *
     * \param offset Offset of the plane
     *
     */
    Plane(const Eigen::Vector3d & normal, double offset);

    /** Constructor for a moving plane
     *
     * \param normal Normal of the plane
     *
     * \param point Point on the plane
     *
     */
    Plane(const Eigen::Vector3d & normal, const Eigen::Vector3d & point);

    /** Set a direct dependency to the outputs of the integrator
     *
     * Such an integrator should update the plane in its own update method
     */
    template<typename S, typename EnumO>
    void setIntegrator(std::shared_ptr<S> integrator, EnumO oPosition, EnumO oVelocity, EnumO oAcceleration)
    {
      addDirectDependency(Output::Position, integrator, oPosition);
      addDirectDependency(Output::Velocity, integrator, oVelocity);
      addDirectDependency(Output::Acceleration, integrator, oAcceleration);
    }

    /** Set a direct dependency to the outputs of the integrator
     *
     * This variant is probably preferable if the integrator creates the plane object
     *
     * Such an integrator should update the plane in its own update method
     */
    template<typename S, typename EnumO,
      typename std::enable_if<std::is_base_of<tvm::graph::abstract::Outputs, S>::value, int>::type = 0>
    void setIntegrator(S & integrator, EnumO oPosition, EnumO oVelocity, EnumO oAcceleration)
    {
      addDirectDependency(Output::Position, integrator, oPosition);
      addDirectDependency(Output::Velocity, integrator, oVelocity);
      addDirectDependency(Output::Acceleration, integrator, oAcceleration);
    }

    /** Change the plane normal and point
     *
     * Triggers offset computation
     */
    void position(const Eigen::Vector3d & normal, const Eigen::Vector3d & point);

    /** Change the plane normal and offset */
    void position(const Eigen::Vector3d & normal, double offset);

    /** Change the point and normal's speeds */
    void velocity(const Eigen::Vector3d & nDot, const Eigen::Vector3d & speed);

    /** Change the point and normal's acceleration */
    void acceleration(const Eigen::Vector3d & nDotDot, const Eigen::Vector3d & accel);

    /** Access the normal */
    const Eigen::Vector3d & normal() const { return normal_; }

    /** Access the offset */
    double offset() const {return offset_; }

    /** Access the point */
    const Eigen::Vector3d & point() const { return point_; }

    /** Access the normal derivative */
    const Eigen::Vector3d & normalDot() const { return normalDot_; }

    /** Access the point's speed */
    const Eigen::Vector3d & speed() const { return speed_; }

    /** Access the normal second derivative */
    const Eigen::Vector3d & normalDotDot() const { return normalDotDot_; }

    /** Access the point's acceleration */
    const Eigen::Vector3d & acceleration() const { return accel_; }
  private:
    Eigen::Vector3d normal_;
    double offset_;
    Eigen::Vector3d point_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normalDot_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d speed_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normalDotDot_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_ = Eigen::Vector3d::Zero();
  };

  using PlanePtr = std::shared_ptr<Plane>;

} // namespace geometry

} // namespace tvm
