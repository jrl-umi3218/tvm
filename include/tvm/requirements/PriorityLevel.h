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
#include <tvm/requirements/abstract/SingleSolvingRequirement.h>

namespace tvm
{

namespace requirements
{

  /** This class represents the priority level of a constraint*/
  class TVM_DLLAPI PriorityLevel : public abstract::SingleSolvingRequirement<int>
  {
  public:
    /** Priority level p>=0*/
    PriorityLevel(int p=0);
  };

}  // namespace requirements

}  // namespace tvm
