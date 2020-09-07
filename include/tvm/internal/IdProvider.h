/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <mutex>

namespace tvm
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
} // namespace internal

} // namespace tvm
