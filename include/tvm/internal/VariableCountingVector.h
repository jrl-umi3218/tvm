/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/Space.h>
#include <tvm/VariableVector.h>
#include <tvm/internal/RangeCounting.h>
#include <tvm/utils/internal/map.h>

namespace tvm::internal
{
/** This class adds a counting logic over a VariableVector, that allows to add
 * and remove subvariables without constraints, and tracks exactly how many
 * time a (part of) variable was added and removed.
 * This is in particular useful for computing the variables of a problem, where
 * different part of a same variables can be added by different functions.
 *
 * This class is used in places where one must deduce a VariableVector from the
 * union of several variables or collections of variables with no constraints on
 * how these variables relate to one another (in particular intersecting
 * subvariables can be used).
 * The process is then:
 *  - perform all the add/remove
 *  - get the resulting VariableVector
 */
class TVM_DLLAPI VariableCountingVector
{
public:
  /** \param split If true, variable parts who do not appear with the same count
   * or were not added at the same time are separated.
   * For example, if x is a variable of size 8, adding x[0:3] then x[4:7] will
   * yield a single variable (x) when split = false but two variables (x[0:3],
   * x[4:7]) otherwise.
   * Adding x[0:4] then x[3:7] will also yield a single variable if split = false
   * but 3 variables (x[0:2], x[3:4] and x[5:7]) otherwise.
   */
  VariableCountingVector(bool split = false) : split_(split) {}
  /** Add a variable. Return \c true if this changes the vector.*/
  bool add(VariablePtr v);
  void add(const VariableVector & v);
  /** Remove a variable. Return \c true if this changes the vector.*/
  bool remove(const Variable & v);
  void remove(const VariableVector & v);

  void clear();

  void set(const VectorConstRef & val);

  /** Return the vector of variables resulting from the different add and remove. */
  const VariableVector & variables() const;

  /** Return a vector mirroring variables() where simple()[i] is \a true when
   * variables()[i] appears because of a single initial add (even if the same
   * variable was subsequently added/removed several times after or subvariables
   * of it were added/removed).
   *
   * This is conservative in the sense that some combinations of add/remove that
   * would end up with a variable abiding the above criterion could be flagged as
   * \a false. When using only add, this is exact, though.
   *
   * \note Only for VariableCountingVector with split = false
   */
  const std::vector<uint8_t> simple() const;

  bool split() const { return split_; }

  /** Check that none of the added variables intersect.*/
  bool isDisjointUnion();

private:
  struct SpaceRangeCounting
  {
    tvm::internal::RangeCounting mSize_;
    tvm::internal::RangeCounting rSize_;
    tvm::internal::RangeCounting tSize_;

    bool add(const Space & start, const Space & dim);
    bool remove(const Space & start, const Space & dim);
    bool empty() const { return mSize_.empty(); }
    int maxCount() const { return mSize_.maxCount(); }
  };

  void update() const;

  tvm::utils::internal::map<Variable *, std::pair<SpaceRangeCounting, size_t>> count_;
  bool split_;
  mutable bool upToDate_ = false;
  mutable VariableVector variables_;
  mutable std::vector<uint8_t> simple_;
};
} // namespace tvm::internal
