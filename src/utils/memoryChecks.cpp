/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/utils/memoryChecks.h>

#include <queue>

namespace tvm::utils
{

std::queue<bool> malloc_is_allowed_override_;

bool is_malloc_allowed()
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  return Eigen::internal::is_malloc_allowed();
#else
  return true;
#endif
}

bool set_is_malloc_allowed(bool allow)
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  return Eigen::internal::set_is_malloc_allowed(allow);
#else
  return allow;
#endif
}

void override_is_malloc_allowed(bool v)
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  malloc_is_allowed_override_.push(is_malloc_allowed());
  if(malloc_is_allowed_override_.back() != v)
    Eigen::internal::set_is_malloc_allowed(v);
#endif
}

void restore_is_malloc_allowed()
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  if(Eigen::internal::is_malloc_allowed() != malloc_is_allowed_override_.back())
    Eigen::internal::set_is_malloc_allowed(malloc_is_allowed_override_.back());
  malloc_is_allowed_override_.pop();
#endif
}

} // namespace tvm::utils