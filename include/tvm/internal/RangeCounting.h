/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <tvm/Range.h>

#include <list>

namespace tvm::internal
{
/** Keep track of the union and (set) subtraction of integer ranges, accounting
 * for the number of times a number was added.
 * For example adding the set {1,2,3,4} (represented by Range(1,4)) and the set
 * {3,4,5,6} (Range(3,4)) will result in a single range, Range(1,6). But
 * internally, it is known that 3 and 4 appeared twice so that removing {2,3,4}
 * (Range(2,3)) will result in a set {1,3,4,5,6}, i.e. the ranges Range(1,1) and
 * Range(3,4).
 *
 *
 * Internally, the ranges are represented by their limits, following the idea in
 * https://stackoverflow.com/a/32869470/11611648, e.g. Range(1,4) = {1,2,3,4} is
 * represented by {(1,+), (5,-)} where the lower limit (opening, denoted by +)
 * is included and the upper limit (closing, denoted by -) is excluded.
 * A list of such limits is kept as a representation of the current state.
 * The number of appearances of an integer is encoded by the number of + and -
 * appearing to reach that number from the start of this list. For example, if
 * the list is {(1,+), (3,+), (5,-), (6,+), (7,-), (7,-), (9,+), (10,-)}, all
 * number up to 0 don't appear, 1 and appear once, 3 and 4 appear twice, 5
 * appears once, 6 appears twice, 7 and 8 don't appear, 9 appears once and all
 * numbers from 10 don't appear. The resulting range representation is
 * { Range(1,7), Range(9,1) }.
 */
class TVM_DLLAPI RangeCounting
{
public:
  /** The lower (included) or upper (excluded) limit of a range*/
  struct Limit
  {
    Limit(int i, bool lower) : i_(i), lower_(lower) {}

    int i_;      // Value of the limit
    bool lower_; // Whether it is a lower or upper limit

    bool operator<(const Limit & other) const { return i_ < other.i_ || (i_ == other.i_ && lower_ && !other.lower_); }

    friend std::ostream & operator<<(std::ostream & os, const Limit & lim)
    {
      os << "(" << lim.i_ << ", " << (lim.lower_ ? "+" : "-") << ")";
      return os;
    }
  };

  /** Add a range to the counting. Return true if this changes the outuput of
   * \c ranges(false) (it always changes the output of \c range(true) ).
   */
  bool add(const Range & r);
  /** Remove a range from the counting. Return true if this changes the outuput
   * of \c ranges() (it always changes the output of \c range(true) ).
   */
  bool remove(const Range & r);

  bool empty() const { return limits_.size() == 0; }

  /** Get a representation of number appearing as a list of ranges.*/
  const std::vector<Range> & ranges() const;
  /** Get the underlying representation as a list of limits.*/
  const std::list<Limit> & limits() const;

private:
  /** Set recompute_ = recompute || change and return change. */
  bool recompute(bool change);

  std::list<Limit> limits_;              // The representation of the current state.
  mutable bool recompute_ = false;       // Need to recompute intervals_. Used for lazy evaluation.
  mutable std::vector<Range> intervals_; // The range representation of the current state, reevaluated
                                         // if necessary when accessed (lazy evaluation).
};
} // namespace tvm::internal