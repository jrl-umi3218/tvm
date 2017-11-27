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
#include <tvm/defs.h>

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <string>

namespace tvm
{
  /** Description of a variable space, and factory for Variable.
   *
   * The space can have up to 3 different sizes, although Euclidean spaces will
   * have only one :
   * - the size of the space as a manifold,
   * - the size of the vector needed to represent one variable (point) in this
   *   space (representation space, rsize),
   * - the size of the vector needed to represent a derivative (velocity,
   *   acceleration, ...) of this variable (tangent space, tsize).
   *
   * Here are a few examples:
   * - R^n has a size, rsize and tsize of n,
   * - SO(3), the 3d rotation space, when represented by quaternions, has a
   *   size of 3, a rsize of 4 and a tsize of 3,
   * - S(2), the sphere in dimension 3 has a size of 2, and rsize and tsize of 3.
   */
  class TVM_DLLAPI Space
  {
  public:
    /** Constructor for an Euclidean space
      *
      * /param size size of the space
      */
    Space(int size);
    /** Constructor for a manifold with tsize = size
      *
      * /param size size of the space
      * /param representationSize size of the vector needed to represent a variable
      */
    Space(int size, int representationSize);
    /** Constructor for a manifold where tsize != size
      *
      * /param size size of the space
      * /param representationSize size of the vector needed to represent a variable
      * /param tangentRepresentationSize size of the vector needed to represent a derivative
      */
    Space(int size, int representationSize, int tangentRepresentationSize);

    /** Factory function to create a variable.*/
    std::unique_ptr<Variable> createVariable(const std::string& name) const;

    /** Size of the space (as a manifold) */
    int size() const;
    /** Size of the vector needed to represent a variable in this space.*/
    int rSize() const;
    /** Size of the vector needed to represent a derivative in this space.*/
    int tSize() const;
    bool isEuclidean() const;

  private:
    int mSize_;   //size of this space (as a manifold)
    int rSize_;   //size of a vector representing a point in this space
    int tSize_;   //size of a vector representing a velocity in this space

    friend bool operator==(const Space&, const Space&);
  };

  inline bool operator==(const Space& s1, const Space& s2)
  {
    return s1.mSize_ == s2.mSize_ && s1.rSize_ == s2.rSize_ && s1.tSize_ == s2.tSize_;
  }
}  // namespace tvm
