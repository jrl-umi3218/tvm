/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>
#include <tvm/constraint/abstract/LinearConstraint.h>
#include <tvm/hint/internal/Substitutions.h>

#include <algorithm>

namespace tvm
{

namespace hint
{

namespace internal
{
/** Return true if \p s is using (part of) variables substituted by \p t.*/
static bool dependsOn(const Substitution & s, const Substitution & t)
{
  for(const auto & x : t.variables())
  {
    for(const auto & c : s.constraints())
    {
      if(c->variables().intersects(*x))
      {
        return true;
      }
    }
  }
  return false;
}

void tvm::hint::internal::Substitutions::add(const Substitution & s)
{
  auto i = dependencies_.addNode();
  assert(i == substitutions_.size());
  substitutions_.push_back(s);

  for(size_t j = 0; j < substitutions_.size(); ++j)
  {
    if(dependsOn(substitutions_[j], s))
    {
      dependencies_.addEdge(j, i);
    }
    if(dependsOn(s, substitutions_[j]))
    {
      dependencies_.addEdge(i, j);
    }
  }
}

bool tvm::hint::internal::Substitutions::remove(LinearConstraintPtr cstr)
{
  bool changed = false;
  for(size_t i = 0; i < substitutions_.size();)
  {
    const auto & s = substitutions_[i];
    const auto & cs = s.constraints();
    auto it = std::find_if(cs.begin(), cs.end(), [&](const auto & c) { return c.get() == cstr.get(); });
    if(it == cs.end())
    {
      ++i;
      continue;
    }
    changed = true;
    dependencies_.removeNode(i);
    substitutions_.erase(substitutions_.begin() + i);
  }
  return changed;
}

const std::vector<Substitution> & Substitutions::substitutions() const { return substitutions_; }

bool Substitutions::uses(LinearConstraintPtr c) const
{
  for(const auto & s : substitutions_)
  {
    for(const auto & cstr : s.constraints())
    {
      if(c == cstr)
        return true;
    }
  }
  return false;
}

void Substitutions::finalize()
{
  // Detect interdependent substitutions (this corresponds to strongly connected
  // components of the dependency graph).
  tvm::graph::internal::DependencyGraph g;
  std::vector<std::vector<size_t>> scc;
  std::tie(scc, g) = dependencies_.reduce();

  // Compute the groups of substitutions and the order to carry out the substitutions
  // in each group. Indices in orderedGroups are relative to scc.
  auto orderedGroups = g.groupedOrder();

  // We create a unit for each group
  units_.clear();
  for(const auto & g : orderedGroups)
  {
    units_.emplace_back(substitutions_, scc, g);
  }

  // Retrieve all the variables, functions and constraints
  variables_.clear();
  varSubstitutions_.clear();
  additionalConstraints_.clear();
  otherVariables_.clear();
  for(const auto & u : units_)
  {
    const auto & x = u.variables();
    const auto & f = u.variableSubstitutions();
    const auto & z = u.additionalVariables();
    const auto & c = u.additionalConstraints();
    const auto & y = u.otherVariables();
    variables_.insert(variables_.end(), x.begin(), x.end());
    varSubstitutions_.insert(varSubstitutions_.end(), f.begin(), f.end());
    for(auto & zi : z)
    {
      if(zi->size() > 0)
      {
        additionalVariables_.push_back(zi);
      }
    }
    for(auto & ci : c)
    {
      if(ci->size() > 0)
      {
        additionalConstraints_.push_back(ci);
      }
    }
    for(auto & yi : y)
    {
      if(std::find(otherVariables_.begin(), otherVariables_.end(), yi) == otherVariables_.end())
      {
        otherVariables_.push_back(yi);
      }
    }
  }
}

void Substitutions::updateSubstitutions()
{
  for(auto & u : units_)
  {
    u.update();
  }
}

void Substitutions::updateVariableValues() const
{
  for(size_t i = 0; i < variables_.size(); ++i)
  {
    varSubstitutions_[i]->updateValue();
    variables_[i]->value(varSubstitutions_[i]->value());
  }
}

const std::vector<VariablePtr> & Substitutions::variables() const { return variables_; }

const std::vector<std::shared_ptr<function::BasicLinearFunction>> & Substitutions::variableSubstitutions() const
{
  return varSubstitutions_;
}

const std::vector<VariablePtr> & Substitutions::additionalVariables() const { return additionalVariables_; }

const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> & Substitutions::additionalConstraints() const
{
  return additionalConstraints_;
}

const std::vector<VariablePtr> & Substitutions::otherVariables() const { return otherVariables_; }

VariableVector Substitutions::substitute(const VariablePtr & x) const
{
  auto it = std::find(variables_.begin(), variables_.end(), x);
  if(it == variables_.end()) // no substitution of var
  {
    VariableVector v({x});
    return v;
  }
  else // substitution of var
  {
    const auto & f = varSubstitutions_[static_cast<size_t>(it - variables_.begin())];
    return f->variables();
  }
}

} // namespace internal

} // namespace hint

} // namespace tvm
