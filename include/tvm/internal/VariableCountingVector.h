/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/api.h>
#include <tvm/defs.h>

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
 */
class TVM_DLLAPI VariableCountingVector
{
public:
  /** Add a variable. Return \c true if this changes the vector.*/
  bool add(VariablePtr v);
  void add(const VariableVector & v);
  /** Remove a variable. Return \c true if this changes the vector.*/
  bool remove(const Variable & v);
  void remove(const VariableVector & v);

  void value(const VectorConstRef & val);

  /** Return the vector of variables resulting from the different add and remove.
   *
   * \warning This vector is not meant to be differentiated. This could cause
   * troubles with non-Euclidean variables.
   *
   * \internal see comment in code.
   */
  const VariableVector & variables() const;

private:
  tvm::utils::internal::map<Variable *, tvm::internal::RangeCounting> count_;
  mutable bool upToDate_ = false;
  mutable VariableVector variables_;
};
} // namespace tvm::internal
