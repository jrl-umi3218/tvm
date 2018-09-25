#pragma once

/* Copyright 2018 CNRS-UM LIRMM, CNRS-AIST JRL
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

#include <map>

namespace tvm
{

namespace utils
{

namespace internal
{
  template<typename ObjWithId>
  class IdComparator
  {
  public:
    using T = typename std::decay<typename std::remove_pointer<ObjWithId>::type>::type;
    bool operator() (const T* const lhs, const T* const rhs) const
    {
      return lhs->id() < rhs->id();
    }

    bool operator() (const T& lhs, const T& rhs) const
    {
      return lhs.id() < rhs.id();
    }
  };

  template<typename KeyWithId, typename Value>
  using mapOnId = std::map<KeyWithId, Value, IdComparator<KeyWithId> >;

}  // namespace internal

}  // namespace utils

}  // namespace tvm