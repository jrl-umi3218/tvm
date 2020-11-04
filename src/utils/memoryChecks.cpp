/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/utils/memoryChecks.h>

#include <stack>

namespace tvm::utils
{

std::stack<bool> malloc_is_allowed_override_;

bool is_malloc_allowed_()
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  return Eigen::internal::is_malloc_allowed();
#else
  return true;
#endif
}

bool set_is_malloc_allowed_(bool allow)
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  return Eigen::internal::set_is_malloc_allowed(allow);
#else
  return allow;
#endif
}

void override_is_malloc_allowed_(bool allow)
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  malloc_is_allowed_override_.push(Eigen::internal::is_malloc_allowed());
  if(malloc_is_allowed_override_.top() != allow)
    Eigen::internal::set_is_malloc_allowed(allow);
#endif
}

void restore_is_malloc_allowed_()
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
  assert(malloc_is_allowed_override_.size() > 0
         && "restore_is_malloc_allowed called too many times compared to override_is_malloc_allowed.");
  if(Eigen::internal::is_malloc_allowed() != malloc_is_allowed_override_.top())
    Eigen::internal::set_is_malloc_allowed(malloc_is_allowed_override_.top());
  malloc_is_allowed_override_.pop();
#endif
}

} // namespace tvm::utils