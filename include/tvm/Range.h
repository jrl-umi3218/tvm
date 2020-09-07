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
  Range(int s, int d) : start(s), dim(d) {}
  int start;
  int dim;

  bool operator==(const Range & other) const { return this->dim == other.dim && this->start == other.start; }

  bool operator!=(const Range & other) const { return !operator==(other); }
};

} // namespace tvm
