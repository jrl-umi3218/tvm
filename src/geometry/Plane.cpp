/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/geometry/Plane.h>

namespace tvm
{

namespace geometry
{

Plane::Plane(const Eigen::Vector3d & normal, double offset) { position(normal, offset); }

Plane::Plane(const Eigen::Vector3d & normal, const Eigen::Vector3d & point) { position(normal, point); }

void Plane::position(const Eigen::Vector3d & normal, double offset)
{
  normal_ = normal;
  offset_ = offset;
}

void Plane::position(const Eigen::Vector3d & normal, const Eigen::Vector3d & point)
{
  normal_ = normal;
  point_ = point;
  offset_ = -normal_.dot(point_);
}

void Plane::velocity(const Eigen::Vector3d & nDot, const Eigen::Vector3d & sp)
{
  normalDot_ = nDot;
  speed_ = sp;
}

void Plane::acceleration(const Eigen::Vector3d & nDDot, const Eigen::Vector3d & ac)
{
  normalDotDot_ = nDDot;
  accel_ = ac;
}

} // namespace geometry

} // namespace tvm
