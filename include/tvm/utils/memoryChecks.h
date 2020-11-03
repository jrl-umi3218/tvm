/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <Eigen/Core>

namespace tvm::utils
{

#ifdef NDEBUG
inline TVM_DLLAPI bool is_malloc_allowed() { return true; }
inline TVM_DLLAPI bool set_is_malloc_allowed(bool b) { return b; }
inline TVM_DLLAPI void override_is_malloc_allowed(bool) {}
inline TVM_DLLAPI void restore_is_malloc_allowed() {}
#else
/** Allow or disallow dynamic allocation in Eigen operations. */
TVM_DLLAPI bool is_malloc_allowed();
TVM_DLLAPI bool set_is_malloc_allowed(bool allow);
TVM_DLLAPI void override_is_malloc_allowed(bool v);
TVM_DLLAPI void restore_is_malloc_allowed();
#endif

#define TVM_ALLOW_EIGEN_MALLOC(x)             \
tvm::utils::override_is_malloc_allowed(true); \
x;                                            \
tvm::utils::restore_is_malloc_allowed();

} // namespace tvm::utils