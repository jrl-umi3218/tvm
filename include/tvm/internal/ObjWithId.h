/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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
  ObjWithId(const ObjWithId &) = delete;
  ObjWithId(ObjWithId&& other): id_(other.id_) { other.id_ = -1; }
  ObjWithId & operator=(const ObjWithId &) = delete;
  ObjWithId & operator=(ObjWithId&& other) {id_ = other.id_; other.id_ = -1; return *this; }

  int id() const { return id_; }

protected:
  ObjWithId() : id_(ObjWithId::idProvider_.makeId()) {}

private:
  static IdProvider idProvider_;
  int id_;
};

} // namespace internal

} // namespace tvm
