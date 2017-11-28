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

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <string>

namespace tvm
{
  class TVM_DLLAPI Range
  {
  public:
    Range() : start(0), dim(0) {}
    Range(int s, int d) : start(s), dim(d) {}
    int start;
    int dim;
  };

  inline bool operator==(const Range& r1, const Range& r2)
  {
    return r1.dim == r2.dim && r1.start == r2.start;
  }
}  // namespace tvm


