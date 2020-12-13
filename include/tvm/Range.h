/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace tvm
{
/** A pair \p (start, dim) representing the integer range from \p start
 * (included) to \p start+dim (excluded).
 */
class TVM_DLLAPI Range
{
public:
  Range() : start(0), dim(0) {}
  Range(int s, int d) : start(s), dim(d) { assert(d >= 0); }
  int start;
  int dim;

  /** First integer not in the range*/
  int end() const { return start + dim; }

  bool operator==(const Range & other) const { return this->dim == other.dim && this->start == other.start; }

  bool operator!=(const Range & other) const { return !operator==(other); }

  /** Return true if \p i is contained in the range.*/
  bool contains(int i) const { return start <= i && i < end(); }

  /** Return true if \p other is contained in the range.
   *
   * Empty ranges are considered contained if their start is in the range.
   */
  bool contains(const Range & other) const { return this->contains(other.start) && this->end() >= other.end(); }

  /** Return true if both range intersects. */
  bool intersects(const Range & other) const
  {
    return this->contains(other.start) || other.contains(this->start);
  }

  /** Return the range of other within this Range.
   *
   * e.g Range(3,8).relativeRange(5,2) returns Range(2,2)
   */
  Range relativeRange(const Range & other) const
  {
    assert(this->contains(other));
    return {other.start - this->start, other.dim};
  }
};

} // namespace tvm
