/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>

#include <tvm/utils/internal/map.h>

namespace tvm::utils::internal
{
/** Dummy struct for selecting overload*/
struct with_sub
{};

template<typename T>
struct slice_traits
{};

/** A map with Variables as key, that can handle the notion of subvariables.
 *
 * \tparam T type of the object stored by the map
 * \tparam Slicer Given a pair (x,t) in the map with x a variable and t a stored
 * object, this helper structure role is to specify how to take the part of t
 * corresponding to a subvariable y of x, where the relation between y and x is
 * given as a tvm::Range. We call this part a \a slice.
 * This structure must provide the following definitions:
 *  - \c Type, the type of a non-constant slice
 *  - \c ConstType, the type of a constant slice
 *  - \code static Type get(T & t, const Range & r) \endcode, a function to get
 * the slice of \p t corresponding to \p r (non-const case)
 *  - \code static ConstType get(const T & t, const Range & r)\endcode, same
 * function for the constant case.
 * \tparam tSize If \c false, use the range of the variable x. If \c true, use
 * the size of the derivative.
 */
template<typename T, typename Slicer = slice_traits<T>, bool tSize = false>
class MapWithVariableAsKey : public tvm::utils::internal::map<const Variable * const, T>
{
public:
  using Base = tvm::utils::internal::map<const Variable * const, T>;
  using Base::at;

  typename Slicer::Type at(const typename Base::key_type & key, with_sub);
  typename Slicer::ConstType at(const typename Base::key_type & key, with_sub) const;
};

template<typename T, typename Slicer, bool tSize>
inline typename Slicer::Type MapWithVariableAsKey<T, Slicer, tSize>::at(const typename Base::key_type & key, with_sub)
{
  auto it = Base::find(key);
  if(it != Base::end())
  {
    // Key variable is directly in the map
    return it->second;
  }
  else
  {
    // Key variable is not directly in the map, we test if it is a subvariable of one of the map variables.
    for(auto it = Base::begin(); it != Base::end(); ++it)
    {
      if(it->first->contains(*key))
      {
        if((*it->first) == (*key))
        {
          return it->second;
        }
        else
        {
          Range r;
          if constexpr(tSize)
            r = it->first->tSubvariableRange().relativeRange(key->tSubvariableRange());
          else
            r = it->first->subvariableRange().relativeRange(key->subvariableRange());
          return Slicer::get(it->second, r);
        }
      }
    }
  }
  // We didn't find the key in the map
  throw std::out_of_range("[MapWithVariableAsKey::at] Variable " + key->name() + " is not part of this map.");
}

template<typename T, typename Slicer, bool tSize>
inline typename Slicer::ConstType MapWithVariableAsKey<T, Slicer, tSize>::at(const typename Base::key_type & key,
                                                                             with_sub) const
{
  auto it = Base::find(key);
  if(it != Base::end())
  {
    // Key variable is directly in the map
    return it->second;
  }
  else
  {
    // Key variable is not directly in the map, we test if it is a subvariable of one of the map variables.
    for(auto it = Base::begin(); it != Base::end(); ++it)
    {
      if(it->first->contains(*key))
      {
        if((*it->first) == (*key))
        {
          return it->second;
        }
        else
        {
          Range r;
          if constexpr(tSize)
            r = it->first->tSubvariableRange().relativeRange(key->tSubvariableRange());
          else
            r = it->first->subvariableRange().relativeRange(key->subvariableRange());
          return Slicer::get(it->second, r);
        }
      }
    }
  }
  // We didn't find the key in the map
  throw std::out_of_range("[MapWithVariableAsKey::at] Variable " + key->name() + " is not part of this map.");
}

} // namespace tvm::utils::internal
