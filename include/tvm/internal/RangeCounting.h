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
 * This class is mostly meant as a utility for tvm::interal::VariableCountingVector.
 * The idea is that tvm::VariableVector is not keeping track of how many time a
 * part of a variable has been added or removed, and has limitations when it
 * comes to adding or removing subvariable. This class can be used on the ranges
 * of subvariables within their supervariable to keep track of the parts present
 * after all the add and remove.
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
 * Compared to the above link, we add also the notion of cut (denoted by | ),
 * that is useful to keep track of the fact that two ranges were stitched
 * together. For example adding {1,2} (Range(1,2)) and {3,4} (Range(3,2))
 * results in {1,2,3,4} (Range(1,4)), but is represented internally by
 * {(1,+), (3,|), (5,-)}.
 * The following rules are enforced for the internal representation:
 *  (1) limits appears in lexicographic order with + < | < -
 *  (2) (i,|) can only appear once for a given i
 *  (3) the sequence (i,+), (i,-) is replaced by (i,|)
 *  (4) the sequence (i,+), (i,|), (i,-) is replaced by (i,|)
 *  (5) the representation starts with (i,+) and finish with (j,-) i<j
 *  (6) cuts can't appear at depth 0
 */
class TVM_DLLAPI RangeCounting
{
public:
  /** The lower (included) or upper (excluded) limit of a range*/
  struct Limit
  {
    enum Type
    {
      Lower = -1,
      Cut = 0,
      Upper = 1
    };

    Limit(int i, Type type) : i_(i), type_(type) {}

    int i_;     // Value of the limit
    Type type_; // The type of the limit

    bool operator==(const Limit & other) const { return (i_ == other.i_ && type_ == other.type_); }
    bool operator<(const Limit & other) const { return i_ < other.i_ || (i_ == other.i_ && type_ < other.type_); }
    bool operator<=(const Limit & other) const { return i_ < other.i_ || (i_ == other.i_ && type_ <= other.type_); }
    bool operator>(const Limit & other) const { return i_ > other.i_ || (i_ == other.i_ && type_ > other.type_); }
    bool operator>=(const Limit & other) const { return i_ > other.i_ || (i_ == other.i_ && type_ >= other.type_); }

    friend std::ostream & operator<<(std::ostream & os, const Limit & lim)
    {
      os << "(" << lim.i_ << ", " << ((lim.type_ == Lower) ? "+" : ((lim.type_ == Cut) ? "|" : "-")) << ")";
      return os;
    }
  };

  /** Add a range to the counting. Return true if this changes the output of
   * \c ranges(false) (it always changes the output of \c range(true) ).
   */
  bool add(const Range & r);
  /** Remove a range from the counting. Return true if this changes the output
   * of \c ranges() (it always changes the output of \c range(true) ).
   */
  bool remove(const Range & r);

  bool empty() const { return limits_.size() == 0; }

  /** Get a representation of number appearing as a list of ranges.
   *
   * \param splitOnDepthDiff If true, ranges are split on count differences and
   * on Limit::Cut (i.e. two ranges touching but not overlapping will be returned
   * separately, not merged).
   */
  const std::vector<Range> & ranges(bool splitOncountDiff = false) const;
  /** Get the underlying representation as a list of limits.*/
  const std::list<Limit> & limits() const;

  /** Maximum number of appearances of a number.*/
  int maxCount() const;

private:
  using It = std::list<Limit>::iterator;

  /** Forward \p it to first element in limits_ that would come after \p val.
   * Returns \c true if depth went down to \p depthCut in the process
   */
  bool moveToFirstAfter(const Limit & val, It & it, int & depth, int depthCut = 0) const;

  /** Insert \p val before \p it and perform reductions if necessary to enforce
   * the constraints on limits_. Upon return \p it point to the first element after
   * the one that was inserted, or the one that was changed.
   */
  void insert(const Limit & val, It & it, int & depth);

  /** Set recompute_ = recompute || change and return change. */
  bool recompute(bool change);

  bool isValid() const;

  /** The representation of the current state.
   *
   * The elements are ordered by increasing lexicographic order on (Limit::i, Limit::type).
   * The following constraints are enforced:
   *  - there can't be two consecutive elements with equal Limit::i and type Limit::Cut
   *    (when the situation arises, this elements are merged).
   *  - for a same Limit::i, there can't be a succession of types Lower, Cut, Upper. Those
   *    are reduced to a single Cut.
   */
  std::list<Limit> limits_;              // The representation of the current state.
  mutable bool recompute_ = false;       // Need to recompute intervals_ without split. Used for lazy evaluation.
  mutable bool recomputeSplit_ = false;  // Need to recompute intervals_ with split. Used for lazy evaluation.
  mutable std::vector<Range> intervals_; // The range representation of the current state, reevaluated
                                         // if necessary when accessed (lazy evaluation).
};
} // namespace tvm::internal
