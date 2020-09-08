/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/robot/Frame.h>

#include <sch/CD/CD_Pair.h>
#include <sch/S_Object/S_Object.h>

namespace tvm
{

namespace robot
{

/** Represent a convex hull attached to a frame
 *
 * Outputs:
 *
 * - Position: keep the convex hull position up-to-date
 *
 */
class TVM_DLLAPI ConvexHull : public graph::abstract::Node<ConvexHull>
{
public:
  SET_OUTPUTS(ConvexHull, Position)
  SET_UPDATES(ConvexHull, Position)

  /** Constructor with a given S_Object instance
   *
   * \param o sch-core object representing the convex hull
   *
   * \param f frame to which the object is attached
   *
   * \param X_f_o \p o position will be set at each iteration to \f$ {}^{o}X_{f} f.position() \f$
   *
   */
  ConvexHull(std::shared_ptr<sch::S_Object> o, FramePtr f, const sva::PTransformd & X_f_o);

  /** Constructor with the path to the convex hull
   *
   * This assumes you are loading an sch::S_Polyhedron object.
   *
   * \param path path to the sch-object
   *
   * \param f frame to which the object is attached
   *
   * \param X_f_o \p o position will be set at each iteration to \f$ {}^{o}X_{f} f.position() \f$
   *
   */
  ConvexHull(const std::string & path, FramePtr f, const sva::PTransformd & X_f_o);

  /** Make a sch::CD_Pair from a second ConvexHull */
  sch::CD_Pair makePair(const ConvexHull & hull) const;

  /** Access the underlying frame (const) */
  const Frame & frame() const;

  /** Access the underlying frame */
  Frame & frame();

private:
  std::shared_ptr<sch::S_Object> o_;
  FramePtr f_;
  sva::PTransformd X_f_o_;

  void updatePosition();
};

using ConvexHullPtr = std::shared_ptr<ConvexHull>;

} // namespace robot

} // namespace tvm
