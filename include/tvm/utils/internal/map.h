/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <functional>
#include <map>
#include <unordered_map>

namespace tvm::utils::internal
{
template<typename ObjWithId>
class IdLess
{
public:
  using T = typename std::decay<typename std::remove_pointer<ObjWithId>::type>::type;
  constexpr bool operator()(const T * const lhs, const T * const rhs) const { return lhs->id() < rhs->id(); }
  constexpr bool operator()(const T & lhs, const T & rhs) const { return lhs.id() < rhs.id(); }
};

template<typename ObjWithId>
class IdEqual
{
public:
  using T = typename std::decay<typename std::remove_pointer<ObjWithId>::type>::type;
  constexpr bool operator()(const T * const lhs, const T * const rhs) const { return lhs->id() == rhs->id(); }
  constexpr bool operator()(const T & lhs, const T & rhs) const { return lhs.id() == rhs.id(); }
};

template<typename ObjWithId>
class HashId
{
public:
  using T = typename std::decay<typename std::remove_pointer<ObjWithId>::type>::type;
  std::size_t operator()(const T * key) const { return std::hash<int>()(key->id()); }
  std::size_t operator()(const T & key) const { return std::hash<int>()(key.id()); }
};

template<typename KeyWithId, typename Value, typename Allocator = std::allocator<std::pair<const KeyWithId, Value>>>
using map = std::map<KeyWithId, Value, IdLess<KeyWithId>, Allocator>;

template<typename KeyWithId, typename Value, typename Allocator = std::allocator<std::pair<const KeyWithId, Value>>>
using unordered_map = std::unordered_map<KeyWithId, Value, HashId<KeyWithId>, IdEqual<KeyWithId>, Allocator>;

} // namespace tvm::utils::internal
