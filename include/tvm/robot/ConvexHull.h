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

    /** Access the underlying frame */
    const Frame & frame() const;
  private:
    std::shared_ptr<sch::S_Object> o_;
    FramePtr f_;
    sva::PTransformd X_f_o_;

    void updatePosition();
  };

  using ConvexHullPtr = std::shared_ptr<ConvexHull>;

} // namespace robot

} // namespace tvm
