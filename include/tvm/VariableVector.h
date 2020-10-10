/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/defs.h>

#include <tvm/internal/MatrixWithProperties.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace tvm
{
/** A vector of variables, with some useful manipulation/analysis functions.
 *
 * One of the main use of this class is to determine variable mapping, i.e.
 * given a vector aggregating all variables, which section of this vector
 * would correspond to a given variable.
 * There is two approaches for that: either build a map with
 * computeMapping or use the method Variable::getMappingIn. The latter
 * uses a cache in Variable in a way that if one invoke Variable::getMappingIn
 * on any variable contained in a VariableVector, the mapping of all other
 * contained variables will be computed and cached. For repeatedly querying
 * the mapping of those variable w.r.t the same VariableVector, this is the
 * fastest option. However it will be slow if querying alternatively
 * mapping w.r.t different VariableVector on the same variable or set of
 * variables.
 *
 * A given variable can only appear once in a vector. Variables appear in the
 * order they were added, ignoring duplicates.
 *
 * FIXME would it make sense to derive from std::vector<std::shared_ptr<Variable>> ?
 */
class TVM_DLLAPI VariableVector
{
public:
  /** Construct an empty vector*/
  VariableVector();
  /** General construction
   *
   * \tparam Can be any list from the following type:
   *   * VariablePtr
   *   * std::unique_ptr<Variable>
   *   * VariableVector
   *   * std::vector<VariablePtr>
   *
   * \param variables A coma-separated list of instances of the above types.
   *   Variables will be added in the order they appear.
   */
  template<typename... VarPtr>
  VariableVector(VarPtr &&... variables);

  /** Add a variable to the vector if not already present.
   *
   * \param v The variable to be added.
   *
   * \returns True if the variable was added, false otherwise.
   */
  bool add(VariablePtr v);
  /** Add a variable to the vector. This version is mostly to be used directly
   * with the output of tvm::Space::createVariable.
   *
   * \param v The variable to be added.
   *
   * \returns True if the variable was added, false otherwise
   *
   * \internal This version is meant to disembiguate between add(VariablePtr v)
   * and add(const VariableVector& variables) when passing a std::unique_ptr<Variable>.
   */
  bool add(std::unique_ptr<Variable> v);
  /** Same as add(VariablePtr), but for adding a vector of variables.*/
  void add(const std::vector<VariablePtr> & variables);
  /** Same as add(VariablePtr), but for adding a vector of variables.*/
  void add(const VariableVector & variables);
  /** Add a variable to the vector, if not already present, and return the index
   * of the variable in the vector whether it was added or not.
   *
   * \param v The variable to be added.
   * \param containingIndex Specify what to return in case \p v is the subvariable
   * of an already present variable. If \c false, return -1 in this case. If 
   * \c true, return the index of the variable containing \p v.
   *
   * \returns Index of variable \p v in the vector. If \p v is a subvariable of an
   * already present variable, returns -1 or the index of the variable depending
   * on \p containingIndex.
   */
  int addAndGetIndex(VariablePtr v, bool containingIndex = false);

  /** Remove a variable from the vector, if present.
   *
   * \param v the variable to be removed.
   *
   * \returns True if the variable was removed, false otherwise.
   *
   * \warning You can only remove a variable logically equal to one present in
   * the vector. Trying to remove a subpart of a variable in the vector will not
   * remove anything, and return \c false.
   */
  bool remove(const Variable & v);
  /** Remove the variable with the given index.
   *
   * \param i Index of the variable to be removed.
   *
   * \throws std::out_of_range if i is smaller than 0 or greater than the
   * number of variables.
   */
  void remove(int i);

  /** Sum of the sizes of all the variables.*/
  int totalSize() const;
  /** Number of variables*/
  int numberOfVariables() const;
  /** Elementwise access*/
  const VariablePtr operator[](int i) const;
  /** whole vector access*/
  const std::vector<VariablePtr> & variables() const;

  /** Get the concatenation of all variables' value, in the order of the
   * variables as given by variables().
   */
  const Eigen::VectorXd & value() const;
  /** Set the value of all variables from a concatenated vector
   *
   * \param val The concatenated value of all the variables, in the order of
   * the variables as given by variables().
   */
  void value(const VectorConstRef & val);
  /** Set the value of all variables to 0.*/
  void setZero();
  /** Compute the mapping for all variables in this vector. The result is
   * stored in each variable and can be queried by Variable::getMappingIn.
   */
  void computeMapping() const;

  /** Compute the mapping for every variabe and return it.*/
  std::map<const Variable *, Range> computeMappingMap() const;
  /** Check if this vector contains variable \p v or not. */
  bool contains(const Variable & v) const;

  /** Find the index of variable \p v in the vector. Returns -1 if \p v is not
   * present.
   */
  int indexOf(const Variable & v) const;

  /** A timestamp, used internally to determine if a mapping needs to be
   * recomputed or not.
   */
  int stamp() const;

  /** Iterator to the first variable. This enable to use VariableVector
   * directly in range-based for loops
   */
  std::vector<VariablePtr>::const_iterator begin() const;
  /** Iterator past the last variable. This enable to use VariableVector
   * directly in range-based for loops
   */
  std::vector<VariablePtr>::const_iterator end() const;

private:
  void add_(VariablePtr v);
  void remove_(std::vector<VariablePtr>::const_iterator it);
  void getNewStamp() const;

  static int counter;

  mutable int stamp_;
  int size_;
  std::vector<VariablePtr> variables_;

  mutable Eigen::VectorXd value_;
};

template<typename... VarPtr>
VariableVector::VariableVector(VarPtr &&... variables) : VariableVector()
{
  // clang-format off
    static_assert((... && (std::is_same_v<typename std::decay_t<VarPtr>, VariablePtr>
                        || std::is_same_v<typename std::decay_t<VarPtr>, std::unique_ptr<Variable>>
                        || std::is_same_v<typename std::decay_t<VarPtr>, VariableVector>
                        || std::is_same_v<typename std::decay_t<VarPtr>, std::vector<VariablePtr>>)));
  // clang-format on
  (add(std::forward<VarPtr>(variables)), ...);
}

/** Get the vector of ndiff-th time derivatives of the variables of the input
 * vector.
 *
 * \param var the variable to be derived
 * \param ndiff the order of the derivation
 *
 * \warning This recreates a vector from scratch each time
 */
VariableVector TVM_DLLAPI dot(const VariableVector & vars, int ndiff = 1);

inline std::vector<VariablePtr>::const_iterator tvm::VariableVector::begin() const { return variables_.begin(); }

inline std::vector<VariablePtr>::const_iterator tvm::VariableVector::end() const { return variables_.end(); }

} // namespace tvm
