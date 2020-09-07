/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/function/abstract/Function.h>
#include <tvm/robot/ConvexHull.h>

namespace tvm
{

namespace robot
{

/** This class implements a collision function for two given objects */
class TVM_DLLAPI CollisionFunction : public function::abstract::Function
{
public:
  SET_UPDATES(CollisionFunction, Value, Velocity, Jacobian, Time, NormalAcceleration)

  /** Constructor
   *
   * \param dt Timestep
   *
   */
  CollisionFunction(Clock & clock);

  /** Add a collision
   *
   * \param ch1 convex hull object for the first object
   *
   * \param ch2 convex hull object for the second object
   *
   */
  void addCollision(ConvexHullPtr ch1, ConvexHullPtr ch2);

  /** Remove all collisions */
  void reset();

protected:
  /* Update functions */
  void updateValue();
  void updateVelocity();
  void updateJacobian();
  void updateTimeDependency();
  void updateNormalAcceleration();

  Clock & clock_;
  uint64_t last_tick_ = 0;

  struct CollisionData
  {
    struct ObjectData
    {
      Eigen::Vector3d nearestPoint_; // In body coordinates
      rbd::Jacobian jac_;
    };
    std::vector<ObjectData> objects_;
    ConvexHullPtr ch_[2];
    sch::CD_Pair pair_;
    Eigen::Vector3d normVecDist_;
    Eigen::Vector3d prevNormVecDist_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d speedVec_ = Eigen::Vector3d::Zero();

    CollisionData();

    CollisionData(CollisionFunction & fn, ConvexHullPtr ch1, ConvexHullPtr ch2);
  };
  std::vector<CollisionData> colls_;

  /** Intermediate computation */
  Eigen::Vector3d closestPoints_[2];
  Eigen::MatrixXd fullJac_;
  Eigen::MatrixXd distJac_;
};

} // namespace robot

} // namespace tvm
