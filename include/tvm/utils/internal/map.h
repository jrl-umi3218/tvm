/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <map>

namespace tvm
{

namespace utils
{

namespace internal
{
template<typename ObjWithId>
class IdComparator
{
public:
  using T = typename std::decay<typename std::remove_pointer<ObjWithId>::type>::type;
  bool operator()(const T * const lhs, const T * const rhs) const { return lhs->id() < rhs->id(); }

  bool operator()(const T & lhs, const T & rhs) const { return lhs.id() < rhs.id(); }
};

template<typename KeyWithId, typename Value, typename Allocator = std::allocator<std::pair<const KeyWithId, Value>>>
using map = std::map<KeyWithId, Value, IdComparator<KeyWithId>, Allocator>;

} // namespace internal

} // namespace utils

} // namespace tvm
