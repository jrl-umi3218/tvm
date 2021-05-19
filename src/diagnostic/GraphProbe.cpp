/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/diagnostic/GraphProbe.h>

#include <iomanip>
#include <iostream>
#include <queue>
#include <string>

namespace
{
using namespace tvm::diagnostic;
using namespace tvm::graph::internal;

std::string indent(int val) { return std::string(val, ' '); }

// shared code for any and none (see below)
template<bool b>
bool check(const std::vector<GraphProbe::OutputVal> & outputVal,
                std::function<bool(const Eigen::MatrixXd &)> select)
{
  for(const auto & v : outputVal)
  {
    if(select(std::get<Eigen::MatrixXd>(v)))
      return b;
  }
  return !b;
}

// any(outputVal, select) returns true if any value v in outputVal is such that select(v) is true, and false otherwise 
constexpr auto any = check<true>;
// none(outputVal, select) returns true if no value v in outputVal is such that select(v) is true, and false otherwise
constexpr auto none = check<false>;

Eigen::MatrixXd transposeIfVector(const Eigen::MatrixXd & M)
{
  if(M.cols() == 1)
    return M.transpose();
  else
    return M;
};

// find the input corresponding to the direct dependency
template<typename InputContainer>
const Log::Input & findInput(const InputContainer & s, const Log::DirectDependency & d)
{
  for(const auto & i : s)
  {
    if(i.owner == d.owner && i.id == d.input && i.source == d.source)
    {
      return i;
    }
  }
  throw std::runtime_error("Input not found");
}

// find the input corresponding to the input dependency
template<typename InputContainer>
const Log::Input & findInput(const InputContainer & s, const Log::InputDependency & d)
{
  for(const auto & i : s)
  {
    if(i.owner == d.owner && i.id == d.input && i.source == d.source)
    {
      return i;
    }
  }
  throw std::runtime_error("Input not found");
}

// find the update corresponding to the output dependency
template<typename UpdateContainer>
const GraphProbe::Update & findUpdate(const UpdateContainer & s, const Log::OutputDependency & d)
{
  for(const auto & u : s)
  {
    if(u.owner == d.owner && u.id == d.update)
    {
      return u;
    }
  }
  throw std::runtime_error("Update not found");
}

// find the output corresponding to the output dependency
const Log::Output & findOutput(const std::set<Log::Output> & s, const Log::DirectDependency & d)
{
  for(const auto & o : s)
  {
    if(o.owner == d.owner && o.id == d.output)
    {
      return o;
    }
  }
  throw std::runtime_error("Output not found");
}

// find the output corresponding to an input
const Log::Output & findOutput(const std::vector<Log::Output> & v, const Log::Input & i)
{
  for(const auto & o : v)
  {
    if(o.owner == i.source && o.id == i.id)
    {
      return o;
    }
  }
  throw std::runtime_error("Output not found");
}

// find the update that is the origin of an internal dependency
template<typename UpdateContainer>
const Log::Update & findFromUpdate(const UpdateContainer & s, const Log::InternalDependency & d)
{
  for(const auto & u : s)
  {
    if(u.owner == d.owner && u.id == d.from)
    {
      return u;
    }
  }
  throw std::runtime_error("Update not found");
}

} // namespace

namespace tvm::diagnostic
{
GraphProbe::GraphProbe(const graph::internal::Log & log) : log_(log) { registerDefault(); }

std::vector<GraphProbe::OutputVal> GraphProbe::listOutputVal(const graph::CallGraph * const g, bool verbose) const
{
  return listOutputVal(log_.subGraph(g).first, verbose);
}

std::vector<GraphProbe::OutputVal> GraphProbe::listOutputVal(const GraphProbe::Output & o, bool verbose) const
{
  return listOutputVal(log_.subGraph(o).first, verbose);
}

std::unique_ptr<GraphProbe::Node> GraphProbe::followUp(const Output & o,
                                                       std::function<bool(const Eigen::MatrixXd &)> select) const
{
  Processed processed(log_);
  return followUp(o, select, processed);
}


std::vector<std::unique_ptr<GraphProbe::Node>> GraphProbe::followUp(
    const graph::CallGraph * const g,
    std::function<bool(const Eigen::MatrixXd &)> select) const
{
  std::vector<std::unique_ptr<Node>> ret;
  Processed processed(log_);

  // First we retrieve the outputs of the call graph
  auto it = log_.graphOutputs_.find(g);
  if(it == log_.graphOutputs_.end())
    throw std::runtime_error("[GraphProbe::followUp]: graph not in log.");

  std::vector<Output> startingPoints;
  for(const auto & p : it->second)
  {
    for(const auto & i : log_.inputs_)
    {
      if(i.owner == p)
      {
        startingPoints.push_back(findOutput(processed.allOutputs, i));
      }
    }
  }

  // Now we process these outputs
  for(const auto & o : startingPoints)
  {
    processed.outputs.clear();
    processed.updates.clear();
    if(any(outputVal(o), select))
      ret.emplace_back(followUp(o, select, processed));
  }

  return ret;
}

void GraphProbe::print(std::ostream & os, const std::unique_ptr<Node> & node) const { print(os, node, 0); }

void GraphProbe::print(std::ostream & os, const std::vector<std::unique_ptr<Node>> & roots) const
{
  for(const auto & r : roots)
  {
    print(os, r);
    os << "\n";
  }
}


GraphProbe::Processed::Processed(const graph::internal::Log & log)
{
  allOutputs = log.outputs_;
  // Adding outputs referred only as source
  for(const auto & i : log.inputs_)
    allOutputs.push_back({i.id, i.name, i.source});
}

const std::type_index & GraphProbe::getPromotedType(const graph::internal::Log::Pointer & p) const
{
  auto it = log_.types_.find(p.value);
  if(it != log_.types_.end())
    return it->second.back();
  else
    return p.type;
}

bool GraphProbe::addOutputVal(std::vector<OutputVal> & ov, const Output & o) const
{
  if(auto it = outputAccessor_.find({getPromotedType(o.owner).hash_code(), o.id}); it != outputAccessor_.end())
  {
    ov.push_back({o, nullptr, it->second(o.owner.value)});
    return true;
  }
  else if(auto it = outputAccessor_.find({o.owner.type.hash_code(), o.id}); it != outputAccessor_.end())
  {
    ov.push_back({o, nullptr, it->second(o.owner.value)});
    return true;
  }
  else if(auto it = varDepOutputAccessor_.find({getPromotedType(o.owner).hash_code(), o.id}); it != varDepOutputAccessor_.end())
  {
    const auto & varMatPair = it->second(o.owner.value);
    for(const auto & p : varMatPair)
      ov.push_back({o, p.first, p.second});
    return true;
  }
  else if(auto it = varDepOutputAccessor_.find({o.owner.type.hash_code(), o.id}); it != varDepOutputAccessor_.end())
  {
    const auto & varMatPair = it->second(o.owner.value);
    for(const auto & p : varMatPair)
      ov.push_back({o, p.first, p.second});
    return true;
  }
  else
    return false;
}

std::vector<GraphProbe::OutputVal> GraphProbe::outputVal(const Output & o) const
{
  std::vector<OutputVal> ret;
  addOutputVal(ret, o);
  return ret;
}


std::vector<GraphProbe::OutputVal> GraphProbe::listOutputVal(const std::vector<graph::internal::Log::Output> & outputs,
                                                             bool verbose) const
{
  std::vector<OutputVal> ret;

  for(auto o : outputs)
  {
    bool b = addOutputVal(ret, o);
    if(!b && verbose)
    {
      std::cout << "No function to retrieve output" << o.name << " for " << getPromotedType(o.owner).name();
      if(o.owner.type != getPromotedType(o.owner))
        std::cout << " (or" << o.owner.type.name() << ")\n" << std::endl;
      else
        std::cout << "\n";
    }
  }
  return ret;
}
std::unique_ptr<GraphProbe::Node> GraphProbe::followUp(const Output & o,
                                                       std::function<bool(const Eigen::MatrixXd &)> select,
                                                       Processed & processed) const
{
  auto node = std::make_unique<Node>(o);

  auto vals = outputVal(o);
  if(vals.empty())
  {
    std::cout << "No function to retrieve output" << o.name << " for " << getPromotedType(o.owner).name();
    if(o.owner.type != getPromotedType(o.owner))
      std::cout << " (or" << o.owner.type.name() << ")\n" << std::endl;
    else
      std::cout << "\n";
    std::cout << "Stopping search for this node.\n";
  }
  if(processed.outputs[o] || none(vals, select))
    return node;

  processed.outputs[o] = true;
  for(const auto & d : log_.directDependencies_)
  {
    if(d.owner == o.owner && d.output == o.id)
    {
      auto i = findInput(log_.inputs_, d);
      node->children.push_back(followUp(findOutput(processed.allOutputs, i), select, processed));
    }
  }
  for(const auto & d : log_.outputDependencies_)
  {
    if(d.owner == o.owner && d.output == o.id)
    {
      node->children.push_back(followUp(findUpdate(log_.updates_, d), select, processed));
    }
  }
  return node;
}

std::unique_ptr<GraphProbe::Node> GraphProbe::followUp(const Update & u,
                                                       std::function<bool(const Eigen::MatrixXd &)> select,
                                                       Processed & processed) const
{
  auto node = std::make_unique<Node>(u);
  if(processed.updates[u])
    return node;

  processed.updates[u] = true;
  for(const auto & d : log_.inputDependencies_)
  {
    if(d.owner == u.owner && d.update == u.id)
    {
      auto i = findInput(log_.inputs_, d);
      node->children.push_back(followUp(findOutput(processed.allOutputs, i), select, processed));
    }
  }
  for(const auto & d : log_.internalDependencies_)
  {
    if(d.owner == u.owner && d.to == u.id)
    {
      node->children.push_back(followUp(findFromUpdate(log_.updates_, d), select, processed));
    }
  }
  return node;
}

void GraphProbe::print(std::ostream & os, const std::unique_ptr<Node> & node, int depth) const
{
  const std::string s = indent(4 * depth);
  const Eigen::IOFormat fmt(-1, 0, " ", "\n", s + "  ");
  if(node->val.index() == 0)
  {
    const auto & o = mpark::get<Output>(node->val);
    os << s << "* Output " << o.name << " [from " << getPromotedType(o.owner).name() << " (0x" << std::hex
       << o.owner.value
       << std::dec << ")]\n";

    for(const auto & v : outputVal(o))
    {
      const auto & M = std::get<Eigen::MatrixXd>(v);
      const auto & var = std::get<VariablePtr>(v);

      if(var)
      {
        os << s << " - for variable " << var->name() << ":\n" << transposeIfVector(M).format(fmt) << "\n";
      }
      else
      {
        os << transposeIfVector(M).format(fmt) << "\n";
      }
    }
  }
  else
  {
    const auto & u = mpark::get<Update>(node->val);
    os << s << "computed by update " << u.name << " [from " << getPromotedType(u.owner).name() << " (0x" << std::hex
       << u.owner.value
       << std::dec << ")] based on\n";
  }

  for(const auto & c : node->children)
    print(os, c, depth + 1);
}

} // namespace tvm::diagnostic

std::ostream & operator<<(std::ostream & os, const std::vector<tvm::diagnostic::GraphProbe::OutputVal> & vals)
{
  using tvm::VariablePtr;
  using tvm::graph::internal::Log;

  for(const auto & v : vals)
  {
    const auto & p = std::get<Log::Output>(v);
    const auto & M = std::get<Eigen::MatrixXd>(v);
    const auto & var = std::get<VariablePtr>(v);

    if(var)
    {
      os << p.owner.type.name() << "(0x" << std::hex << p.owner.value << std::dec << ") Output " << p.name
         << " for variable " << var->name() << ":\n"
         << transposeIfVector(M) << "\n\n";
    }
    else
    {
      os << p.owner.type.name() << "(0x" << std::hex << p.owner.value << std::dec << ") " << p.name << ":\n"
         << transposeIfVector(M) << "\n\n";
    }
  }
  return os;
}

#include <tvm/constraint/BasicLinearConstraint.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/function/BasicLinearFunction.h>
#include <tvm/function/IdentityFunction.h>
#include <tvm/task_dynamics/Constant.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/Proportional.h>
#include <tvm/task_dynamics/ProportionalDerivative.h>
#include <tvm/task_dynamics/Reference.h>
#include <tvm/task_dynamics/VelocityDamper.h>

namespace tvm::diagnostic
{
void GraphProbe::registerDefault()
{
  registerTVMConstraint<constraint::abstract::Constraint>();
  registerTVMConstraint<constraint::abstract::LinearConstraint>();
  registerTVMConstraint<constraint::internal::LinearizedTaskConstraint> ();
  registerTVMConstraint<constraint::BasicLinearConstraint>();

  registerTVMFunction<function::abstract::Function>();
  registerTVMFunction<function::abstract::LinearFunction>();
  registerAccessor<function::abstract::LinearFunction>(function::abstract::LinearFunction::Output::B,
                                                       &function::abstract::LinearFunction::b);
  registerTVMFunction<function::BasicLinearFunction>();
  registerTVMFunction<function::IdentityFunction>();
  
  registerTVMTaskDynamics<task_dynamics::Constant>();
  registerTVMTaskDynamics<task_dynamics::None>();
  registerTVMTaskDynamics<task_dynamics::Proportional>();
  registerTVMTaskDynamics<task_dynamics::ProportionalDerivative>();
  registerTVMTaskDynamics<task_dynamics::Reference>();
  registerTVMTaskDynamics<task_dynamics::VelocityDamper>();
  registerAccessor<task_dynamics::abstract::TaskDynamicsImpl>(
      task_dynamics::abstract::TaskDynamicsImpl::Output::Value, &task_dynamics::abstract::TaskDynamicsImpl::value);
}
}
