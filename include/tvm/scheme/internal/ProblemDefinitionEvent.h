/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <memory>

namespace tvm
{
class TaskWithRequirements;

namespace scheme::internal
{
/** A class describing a change in a problem definition.*/
class ProblemDefinitionEvent
{
public:
  enum class Type
  {
    WeightChange,
    AnisotropicWeightChange,
    TaskAddition,
    TaskRemoval
  };

  ProblemDefinitionEvent(Type type, TaskWithRequirements * emitter) : type_(type), emitter_(emitter) {}

  Type type() const { return type_; }
  TaskWithRequirements * emitter() const { return emitter_; }

private:
  Type type_;
  TaskWithRequirements * emitter_;
};
} // namespace scheme::internal
} // namespace tvm