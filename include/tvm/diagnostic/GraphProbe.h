/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <tvm/Variable.h>
#include <tvm/diagnostic/internal/probe.h>
#include <tvm/diagnostic/matrix.h>
#include <tvm/graph/CallGraph.h>

#include <mpark/variant.hpp>

#include <Eigen/Core>

#include <iosfwd>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace tvm::diagnostic
{
/** A class to explore the value computed in a call graph.
 *
 * Because the tvm nodes only declare the dependencies between inputs, updates and outputs, but not
 * the way to retrieve an output, it is necessary to register first the method used to access each
 * output of interest. This is done atomically with the methods \c GraphNode::registerAccessor, or
 * at higher level with the methods \c GraphNode::registerTVMFunction, \c GraphNode::registerTVMConstraint
 * and \c GraphNode::registerTVMTaskDynamics. Access methods for base tvm classes are automatically
 * registered in the constructor.
 */
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

  /** Factory returning a function checking if a matrix M has elements whose absolute value is
   * between \p rmin and \p rmax.
   */
  static constexpr auto inRange = [](double rmin, double rmax) {
    return [rmin, rmax](const Eigen::MatrixXd & M) { return hasElemInRange(M, rmin, rmax); };
  };

  /** Function checking if \p contains NaN. */
  static constexpr auto hasNan = [](const Eigen::MatrixXd & M) { return M.hasNaN(); };

  /** Constructor */
  GraphProbe(const graph::internal::Log & log = tvm::graph::internal::Logger::logger().log());

  /** Register a method \p fn with no argument to retrieve the value associated to \p o
   *
   * \tparam T type of the node to which the method is attached. It must be given explicitly
   * if the method is inherited from a base class, otherwise the method will be registered
   * for this base class.
   */
#ifndef _MSC_VER
  template<typename T, typename U, typename EnumOutput, typename Base = T>
  void registerAccessor(
      EnumOutput o,
      const U & (Base::*fn)() const,
      std::function<Eigen::MatrixXd(const U &)> convert = [](const U & u) { return u; });
#else
  template<typename T, typename U, typename EnumOutput>
  void registerAccessor(
      EnumOutput o,
      const U & (T::*fn)() const,
      std::function<Eigen::MatrixXd(const U &)> convert = [](const U & u) { return u; });
#endif

  /** Register a method \p fn taking a variable to retrieve the value associated to \p o
   *
   * \tparam T type of the node to which the method is attached. It must be given explicitly
   * if the method is inherited from a base class, otherwise the method will be registered
   * for this base class.
   */
#ifndef _MSC_VER
  template<typename T, typename U, typename EnumOutput, typename Base = T>
  void registerAccessor(
      EnumOutput o,
      U (Base::*fn)(const Variable &) const,
      std::function<Eigen::MatrixXd(const U &)> convert = [](const U & u) { return u; });
#else
  template<typename T, typename U, typename EnumOutput>
  void registerAccessor(
      EnumOutput o,
      U (T::*fn)(const Variable &) const,
      std::function<Eigen::MatrixXd(const U &)> convert = [](const U & u) { return u; });
#endif

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

  /** Follow up backward the computation graph starting at node \p o, and trimming out branches where the
   * value of an output does not pass the test given by \p select.
   */
  std::unique_ptr<Node> followUp(
      const Output & o,
      std::function<bool(const Eigen::MatrixXd &)> select = [](const Eigen::MatrixXd &) { return true; }) const;

  /** Follow up backward the computation graph starting at the inputs of \p g, and trimming out
   * branches where the value of an output does not pass the test given by \p select.
   */
  std::vector<std::unique_ptr<Node>> followUp(
      const graph::CallGraph * const g,
      std::function<bool(const Eigen::MatrixXd &)> select = [](const Eigen::MatrixXd &) { return true; }) const;

  /** Print in \p os the tree starting at \p root.*/
  void print(std::ostream & os, const std::unique_ptr<Node> & root) const;
  /** Print in \p os the trees starting each at an element of \p root.*/
  void print(std::ostream & os, const std::vector<std::unique_ptr<Node>> & roots) const;

private:
  /** Helper class used in followUp*/
  struct Processed
  {
    Processed() = default;
    Processed(const graph::internal::Log & log);

    std::map<Output, bool> outputs;
    std::map<Update, bool> updates;
    std::vector<Output> allOutputs;
  };

  /** Get the most derived type corresponding to pointer \p p.
   *
   * (This is based on the assumption that the log add the most derived type last)
   */
  const std::type_index & getPromotedType(const graph::internal::Log::Pointer & p) const;

  /** Add the value(s) corresponding to \p o to \p ov.*/
  bool addOutputVal(std::vector<OutputVal> & ov, const Output & o) const;
  /** Return the value(s) corresponding to \p o*/
  std::vector<OutputVal> outputVal(const Output & o) const;
  /** Subfunction for listOutputVal.*/
  std::vector<OutputVal> listOutputVal(const std::vector<graph::internal::Log::Output> & o, bool verbose) const;
  /** Subfunction for followUp*/
  std::unique_ptr<Node> followUp(const Output & o,
                                 std::function<bool(const Eigen::MatrixXd &)> select,
                                 Processed & processed) const;
  /** Subfunction for followUp*/
  std::unique_ptr<Node> followUp(const Update & u,
                                 std::function<bool(const Eigen::MatrixXd &)> select,
                                 Processed & processed) const;

  /** Subfunction for print*/
  void print(std::ostream & os, const std::unique_ptr<Node> & node, int depth) const;

  /** Register the usual tvm class. */
  void registerDefault();

  using OutputKey = std::pair<std::size_t, graph::internal::Log::EnumValue>;
  using VarMatrixPair = std::pair<VariablePtr, Eigen::MatrixXd>;
  const graph::internal::Log & log_;
  std::unordered_map<OutputKey, std::function<Eigen::MatrixXd(uintptr_t)>, internal::PairHasher> outputAccessor_;
  std::unordered_map<OutputKey, std::function<std::vector<VarMatrixPair>(uintptr_t)>, internal::PairHasher>
      varDepOutputAccessor_;
};

#ifndef _MSC_VER
template<typename T, typename U, typename EnumOutput, typename Base>
inline void GraphProbe::registerAccessor(EnumOutput o,
                                         const U & (Base::*fn)() const,
                                         std::function<Eigen::MatrixXd(const U &)> convert)
#else
template<typename T, typename U, typename EnumOutput>
inline void GraphProbe::registerAccessor(EnumOutput o,
                                         const U & (T::*fn)() const,
                                         std::function<Eigen::MatrixXd(const U &)> convert)
#endif
{
#ifndef _MSC_VER
  static_assert(std::is_base_of_v<Base, T>, "Must be called with a method related to T");
#endif
  OutputKey k{std::type_index(typeid(T)).hash_code(), tvm::graph::internal::Log::EnumValue(o)};
  outputAccessor_[k] = [fn, convert](uintptr_t t) { return convert((reinterpret_cast<T *>(t)->*fn)()); };
}

#ifndef _MSC_VER
template<typename T, typename U, typename EnumOutput, typename Base>
inline void GraphProbe::registerAccessor(EnumOutput o,
                                         U (Base::*fn)(const Variable &) const,
                                         std::function<Eigen::MatrixXd(const U &)> convert)
#else
template<typename T, typename U, typename EnumOutput>
inline void GraphProbe::registerAccessor(EnumOutput o,
                                         U (T::*fn)(const Variable &) const,
                                         std::function<Eigen::MatrixXd(const U &)> convert)
#endif
{
#ifndef _MSC_VER
  static_assert(std::is_base_of_v<Base, T>, "Must be called with a method related to T");
#endif
  OutputKey k{std::type_index(typeid(T)).hash_code(), tvm::graph::internal::Log::EnumValue(o)};
  varDepOutputAccessor_[k] = [fn, convert](uintptr_t t) {
    std::vector<VarMatrixPair> ret;
    T * ptr = reinterpret_cast<T *>(t);
    for(const auto & v : ptr->variables())
      ret.emplace_back(v, convert((ptr->*fn)(*v)));
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
  registerAccessor<typename T::Impl>(T::Impl::Output::Value, &T::Impl::value);
}
} // namespace tvm::diagnostic

TVM_DLLAPI std::ostream & operator<<(std::ostream & os,
                                     const std::vector<tvm::diagnostic::GraphProbe::OutputVal> & vals);
