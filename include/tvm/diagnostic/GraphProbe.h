/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>
#include <tvm/api.h>

#include <tvm/diagnostic/internal/probe.h>
#include <tvm/graph/CallGraph.h>

#include <tvm/3rd-party/mpark/variant.hpp>

#include <Eigen/Core>

#include <iosfwd>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace tvm::diagnostic
{
class TVM_DLLAPI GraphProbe
{
public:
  using Output = graph::internal::Log::Output;
  using Update = graph::internal::Log::Update;
  using OutputVal = std::tuple<Output, VariablePtr, Eigen::MatrixXd>;

  struct TVM_DLLAPI Node
  {
    Node(const Output & o) : val(o) {}
    Node(const Update & u) : val(u) {}
    mpark::variant<Output, Update> val;
    std::vector<std::unique_ptr<Node>> children;

    Node & operator=(const Node &) = delete;
    Node(const Node &) = delete;
  };

  GraphProbe(const graph::internal::Log & log);

  /** Register a method \p fn with no argument to retrieve the value associated to \p o
   *
   * \tparam T type of the node to which the method is attached. It must be given explicitly
   * if the method is inherited from a base class, otherwise the method will be registered
   * for this base class.
   */
  template<typename T, typename U, typename EnumOutput>
  void registerAccessor(EnumOutput o, const U & (T::*fn)() const);

  /** Register a method \p fn taking a variable to retrieve the value associated to \p o
   *
   * \tparam T type of the node to which the method is attached. It must be given explicitly
   * if the method is inherited from a base class, otherwise the method will be registered
   * for this base class.
   */
  template<typename T, typename U, typename EnumOutput>
  void registerAccessor(EnumOutput o, U (T::*fn)(const Variable &) const);

  /** Register all methods associated to outputs inherited from tvm::function::abstract::Function */
  template<typename T>
  void registerTVMFunction();

  /** Register all methods associated to outputs inherited from tvm::constraint::abstract::Constraint */
  template<typename T>
  void registerTVMConstraint();

  /** Register all methods associated to outputs inherited from tvm::task_dynamics::abstract::TaskDynamicsImpl */
  template<typename T>
  void registerTVMTaskDynamics();

  /** List all the values of all the outputs present in the call graph \p g (if the associated
   * methods were registered).
   *
   * If \p verbose is \c true, display the outputs for which no methods were registered to retrieve
   * their values.
   */
  std::vector<OutputVal> listOutputVal(const graph::CallGraph * const g, bool verbose = false) const;

  /** List all the values of all the outputs on which \p o depends (recursively) (if the associated
   * methods were registered).
   *
   * If \p verbose is \c true, display the outputs for which no methods were registered to retrieve
   * their values.
   */
  std::vector<OutputVal> listOutputVal(const Output & o, bool verbose = false) const;

  /** Follow up backward the computation graph starting at node \p o, and triming out branches where the
   * value of an output does not pass the test given by \p select.
   */
  std::unique_ptr<Node> followUp(
      const Output & o,
      std::function<bool(const Eigen::MatrixXd &)> select = [](const Eigen::MatrixXd &) { return true; }) const;

  std::vector<std::unique_ptr<Node>> followUp(
      const graph::CallGraph * const g,
      std::function<bool(const Eigen::MatrixXd &)> select = [](const Eigen::MatrixXd &) { return true; }) const;

  /** Print in \p os the tree starting at root \p Node*/
  void print(std::ostream& os, const std::unique_ptr<Node> & node) const;

private:
  struct Processed
  {
    std::map<Output, bool> outputs;
    std::map<Update, bool> updates;
    std::vector<Output> allOutputs;
  };

  const std::type_index & getPromotedType(const graph::internal::Log::Pointer & p) const;

  bool addOutputVal(std::vector<OutputVal> & ov, const Output & o) const;
  std::vector<OutputVal> listOutputVal(const std::vector<graph::internal::Log::Output> & o, bool verbose) const;
  std::unique_ptr<Node> followUp(const Output & o,
                                 std::function<bool(const Eigen::MatrixXd &)> select,
                                 Processed & processed) const;
  std::unique_ptr<Node> followUp(const Update & u,
                                 std::function<bool(const Eigen::MatrixXd &)> select,
                                 Processed & processed) const;

  void print(std::ostream & os, const std::unique_ptr<Node> & node, int depth) const;

  using OutputKey = std::pair<std::size_t, graph::internal::Log::EnumValue>;
  using VarMatrixPair = std::pair<VariablePtr, Eigen::MatrixXd>;
  const graph::internal::Log & log_;
  std::unordered_map<OutputKey, std::function<Eigen::MatrixXd(uintptr_t)>, internal::PairHasher> outputAccessor_;
  std::unordered_map<OutputKey, std::function<std::vector<VarMatrixPair>(uintptr_t)>, internal::PairHasher>
      varDepOutputAccessor_;
};

template<typename T, typename U, typename EnumOutput>
inline void GraphProbe::registerAccessor(EnumOutput o, const U & (T::*fn)() const)
{
  OutputKey k{std::type_index(typeid(T)).hash_code(), tvm::graph::internal::Log::EnumValue(o)};
  outputAccessor_[k] = [fn](uintptr_t t) { return (reinterpret_cast<T *>(t)->*fn)(); };
}

template<typename T, typename U, typename EnumOutput>
inline void GraphProbe::registerAccessor(EnumOutput o, U (T::*fn)(const Variable &) const)
{
  OutputKey k{std::type_index(typeid(T)).hash_code(), tvm::graph::internal::Log::EnumValue(o)};
  varDepOutputAccessor_[k] = [fn](uintptr_t t) {
    std::vector<VarMatrixPair> ret;
    T * ptr = reinterpret_cast<T *>(t);
    for(const auto & v : ptr->variables())
      ret.emplace_back(v, (ptr->*fn)(*v));
    return ret;
  };
}

template<typename T>
inline void GraphProbe::registerTVMFunction()
{
  registerAccessor<T>(T::Output::Value, &T::value);
  registerAccessor<T>(T::Output::Velocity, &T::velocity);
  registerAccessor<T>(T::Output::Jacobian, &T::jacobian);
  registerAccessor<T>(T::Output::NormalAcceleration, &T::normalAcceleration);
  registerAccessor<T>(T::Output::JDot, &T::JDot);
}

template<typename T>
inline void GraphProbe::registerTVMConstraint()
{
  registerAccessor<T>(T::Output::Value, &T::value);
  registerAccessor<T>(T::Output::Jacobian, &T::jacobian);
  registerAccessor<T>(T::Output::L, &T::l);
  registerAccessor<T>(T::Output::U, &T::u);
  registerAccessor<T>(T::Output::E, &T::e);
}

template<typename T>
inline void GraphProbe::registerTVMTaskDynamics()
{
  registerAccessor<T>(T::Output::Value, &T::value);
}
} // namespace tvm::diagnostic

TVM_DLLAPI std::ostream & operator<<(std::ostream & os,
                                     const std::vector<tvm::diagnostic::GraphProbe::OutputVal> & vals);
