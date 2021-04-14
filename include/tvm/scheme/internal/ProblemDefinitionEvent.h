/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <memory>

namespace tvm
{
class TaskWithRequirements;
namespace hint
{
class Substitution;
}

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
    TaskRemoval,
    SubstitutionAddition,
    SubstitutionRemoval
  };

  template<Type t>
  using EmitterType = std::conditional_t<t <= Type::TaskRemoval, TaskWithRequirements, hint::Substitution>;

  ProblemDefinitionEvent(Type type, const TaskWithRequirements & emitter)
  : type_(type), emitter_(static_cast<void const *>(&emitter))
  {
    assert(type <= Type::TaskRemoval);
  }

  ProblemDefinitionEvent(Type type, hint::Substitution const * emitter)
  : type_(type), emitter_(static_cast<void const *>(emitter))
  {
    assert(type == Type::SubstitutionAddition || type == Type::SubstitutionRemoval);
  }

  ProblemDefinitionEvent(const ProblemDefinitionEvent &) = default;
  ProblemDefinitionEvent(ProblemDefinitionEvent &&) = default;
  ProblemDefinitionEvent & operator=(const ProblemDefinitionEvent &) = default;
  ProblemDefinitionEvent & operator=(ProblemDefinitionEvent &&) = default;

  Type type() const { return type_; }
  void const * const emitter() const { return emitter_; }
  template<Type t>
  const auto & typedEmitter() const
  {
    return *static_cast<EmitterType<t> const *>(emitter_);
  }

private:
  Type type_;
  void const * emitter_;
};
} // namespace scheme::internal
} // namespace tvm