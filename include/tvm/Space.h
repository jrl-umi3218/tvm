/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

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
    /** Predifined space types.*/
    enum class Type
    {
      Euclidean,    /** Euclidean space \f$ \mathbb{R}^n \f$.*/
      SO3,          /** Space of 3d rotations, represented by unit quaternions.*/
      SE3,          /** Space of 3d transformation, represented by a quaternion and a 3d-vector.*/
      Unspecified   /** Non-euclidean space of unknown type.*/
    };

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

    /** Constructor for a given space type
      *
      * /param type type of space
      * /param size size of the space. Only for space types whose size is not fixed.
      */
    Space(Type type, int size=-1);

    /** Factory function to create a variable.*/
    std::unique_ptr<Variable> createVariable(const std::string& name) const;

    /** Size of the space (as a manifold) */
    int size() const;
    /** Size of the vector needed to represent a variable in this space.*/
    int rSize() const;
    /** Size of the vector needed to represent a derivative in this space.*/
    int tSize() const;
    /** Type of the space. */
    Type type() const;
    /** Check if this is an Euclidean space.*/
    bool isEuclidean() const;

    bool operator==(const Space& other) const
    {
      return this->mSize_ == other.mSize_ && this->rSize_ == other.rSize_ && this->tSize_ == other.tSize_;
    }

    bool operator!=(const Space& other) const
    {
      return !operator==(other);
    }

  private:
    int  mSize_;   //size of this space (as a manifold)
    int  rSize_;   //size of a vector representing a point in this space
    int  tSize_;   //size of a vector representing a velocity in this space
    Type type_;    //the space type
  };

}  // namespace tvm
