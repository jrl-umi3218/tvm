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

#include <tvm/api.h>

#include <tvm/internal/IdProvider.h>

namespace tvm
{

namespace internal
{

  /** A class with a unique id.*/
  class TVM_DLLAPI ObjWithId
  {
  public:
    int id() const { return id_; }

  protected:
    ObjWithId() : id_(ObjWithId::idProvider_.makeId()) {}

  private:
    static IdProvider idProvider_;
    int id_;
  };

}  // namespace internal

}  // namespace tvm