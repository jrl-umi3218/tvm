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

#include <mutex>

namespace tvm
{

namespace scheme
{

namespace internal
{
  /** A small helper class to provide unique ids.*/
  class TVM_DLLAPI IdProvider
  {
  public:
    int makeId();
  private:
    std::mutex mutex_;
    int id_ = 0;
  };


  inline int IdProvider::makeId()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return id_++;
  }
}  // namespace internal

}  // namespace scheme

}  // namespace tvm
