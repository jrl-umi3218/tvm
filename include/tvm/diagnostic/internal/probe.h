/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <functional>
#include <typeindex>
#include <utility>

namespace tvm::diagnostic::internal
{
/** Combine the hash value of \p v with the value \p seed
 *
 * Taken from https://stackoverflow.com/a/23860042
 */
template<typename T>
inline void hash_combine(std::size_t & seed, const T & v) noexcept
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/** Functor to hash a pair of object.
 *
 * Adapted from https://stackoverflow.com/a/23860042
 */
struct PairHasher
{
  template<typename T, typename U>
  std::size_t operator()(const std::pair<T, U> & p) const noexcept
  {
    std::size_t hash = 0;
    hash_combine(hash, p.first);
    hash_combine(hash, p.second);
    return hash;
  }
};

/** An helper function to remove a noexcept from the signature of a method.
 * 
 * Useful for the registration of methods in GraphProbe::registerAccessor
 */
template<typename T, typename U, typename... Args>
auto remove_noexcept(U (T::*fn)(Args...) const noexcept)
{
  return static_cast<U (T::*)(Args...) const>(fn);
}
} // namespace tvm::diagnostic::internal

// Specialization
#include <tvm/graph/internal/Log.h>

namespace std
{
#ifdef WIN32
/**  Specialization of std::hash for std::size_t */
template<>
struct hash<std::size_t>
{
  // identity function for this type
  std::size_t operator()(const std::size_t & i) const noexcept { return i; }
};
#endif

/**  Specialization of std::hash for tvm::graph::internal::Log::EnumValue */
template<>
struct hash<tvm::graph::internal::Log::EnumValue>
{
  std::size_t operator()(const tvm::graph::internal::Log::EnumValue & e) const noexcept
  {
    std::size_t hash = 0;
    tvm::diagnostic::internal::hash_combine(hash, e.type);
    tvm::diagnostic::internal::hash_combine(hash, e.value);
    return hash;
  }
};
} // namespace std
