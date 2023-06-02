/* Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/VariableVector.h>
#include <tvm/scheme/internal/MatrixAssignment.h>

namespace tvm::scheme::internal
{
void MatrixAssignment::updateTarget(const AssignmentTarget & target)
{
  assignment.to((target.*getTargetMatrix)(colRange.start, colRange.dim));
}

void MatrixAssignment::updateMapping(const VariableVector & newVar,
                                     const AssignmentTarget & target,
                                     bool updateMatrixTarget)
{
  if(newVar.contains(*x))
  {
    Range newRange = x->getMappingIn(newVar);
    if(newRange != colRange)
    {
      colRange = newRange;
    }
    if(updateMatrixTarget)
    {
      updateTarget(target);
    }
  }
  else
  {
    throw std::runtime_error(
        "[Assignment::MatrixAssignment::updateMapping]: new variables do not include this assignment variable.");
  }
}
} // namespace tvm::scheme::internal
