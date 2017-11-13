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

namespace tvm
{

namespace requirements
{

namespace abstract
{

  /** A class representing the way a constraint has to be solved and how it
    * interacts with other constraints in term of hierarchical and weighted
    * priorities.
    *
    * This is a base class for the sole purpose of conveniency.
    */
  template<typename T>
  class SingleSolvingRequirement
  {
  public:
    const T& value() const { return value_; }
    bool isDefault() const { return default_; }

  protected:
    SingleSolvingRequirement(const T& val, bool isDefault)
      : default_(isDefault), value_(val)
    {}

    /** Is this requirement at it default value*/
    bool default_;

    T value_;
  };

}  // namespace abstract

}  // namespace requirements

}  // namespace tvm
