/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <Eigen/Core>

namespace tvm::utils
{

#ifdef NDEBUG
inline bool is_malloc_allowed() { return true; }
inline bool set_is_malloc_allowed(bool b) { return b; }
inline void override_is_malloc_allowed(bool) {}
inline void restore_is_malloc_allowed() {}
#else
namespace internal
{
TVM_DLLAPI bool is_malloc_allowed_();
TVM_DLLAPI bool set_is_malloc_allowed_(bool allow);
TVM_DLLAPI void override_is_malloc_allowed_(bool v);
TVM_DLLAPI void restore_is_malloc_allowed_();

inline void check_malloc_coherency()
{
#  ifdef EIGEN_RUNTIME_NO_MALLOC
  assert(Eigen::internal::is_malloc_allowed() == is_malloc_allowed_()
         && "Eigen::internal::is_malloc_allowed() at the caller site and tvm::utils::is_malloc_allowed are differing. "
            "This might be due to the fact that Eigen::internal::set_is_alloc_allowed was called independently/instead "
            "of tvm::utils::set_is_malloc_allowed in the current program/library.");
#  endif
}

/** Ensure that Eigen::internal::is_malloc_allowed() and tvm::utils::is_malloc_allowed() are the same.
 *
 * This function is not meant to be called directly (but doing so should not change anything).
 */
inline void enforce_malloc_coherency()
{
#  ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(is_malloc_allowed_());
#  endif
}
} // namespace internal

/** Return whether dynamic allocation of memory in Eigen objects is allowed or not.*/
inline bool is_malloc_allowed()
{
  internal::check_malloc_coherency();
  return internal::is_malloc_allowed_();
}

/** Set whether dynamic allocation of memory in Eigen objects is allowed or not.*/
inline bool set_is_malloc_allowed(bool allow)
{
  internal::check_malloc_coherency();
  internal::set_is_malloc_allowed_(allow);
  internal::enforce_malloc_coherency();
  return allow;
}

/** Record whether dynamic allocation of memory in Eigen objects is allowed or not and set \p allow instead.
 * Records are made with a stack.
 *
 * Any call to this function should be mirrored by a call to restore_is_malloc_allowed().
 * It is advised to call restore_is_malloc_allowed() in the same scope.
 */
inline void override_is_malloc_allowed(bool allow)
{
  internal::check_malloc_coherency();
  internal::override_is_malloc_allowed_(allow);
  internal::enforce_malloc_coherency();
}

/** Restore the latest value recorded by override_is_malloc_allowed that was not already restored.*/
inline void restore_is_malloc_allowed()
{
  internal::check_malloc_coherency();
  internal::restore_is_malloc_allowed_();
  internal::enforce_malloc_coherency();
}
#endif

#define TVM_TEMPORARY_ALLOW_EIGEN_MALLOC(x)     \
  tvm::utils::override_is_malloc_allowed(true); \
  x;                                            \
  tvm::utils::restore_is_malloc_allowed();

} // namespace tvm::utils
