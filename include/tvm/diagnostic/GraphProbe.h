/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <tvm/Variable.h>
#include <tvm/diagnostic/internal/probe.h>
#include <tvm/diagnostic/internal/traits.h>
#include <tvm/diagnostic/matrix.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/internal/MatrixWithProperties.h>

#include <mpark/variant.hpp>

#include <Eigen/Core>

#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
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

  struct TVM_DLLAPI ProbeNode
  {
    ProbeNode(const Output & o) : val(o) {}
    ProbeNode(const Update & u) : val(u) {}
    mpark::variant<Output, Update> val;
    std::vector<std::unique_ptr<ProbeNode>> children;

    ProbeNode & operator=(const ProbeNode &) = delete;
    ProbeNode(const ProbeNode &) = delete;
  };

  /** Factory returning a function checking if a matrix M has elements whose absolute value is
   * between \p rmin and \p rmax.
   *
   * This function can be used as a select function in \c GraphProbe::followUp.
   */
  static constexpr auto absInRange = [](double rmin, double rmax) {
    return [rmin, rmax](const Eigen::MatrixXd & M) { return hasElemAbsInRange(M, rmin, rmax); };
  };

  /** Function checking if \p M contains NaN.
   *
   * This function can be used as a select function in \c GraphProbe::followUp.
   */
  static constexpr auto hasNan = [](const Eigen::MatrixXd & M) { return M.hasNaN(); };

  /** Constructor */
  GraphProbe(const graph::internal::Log & log = tvm::graph::internal::Logger::logger().log());

  /** Register a method \p fn to retrieve the value associated to \p o
   *
   * The method either takes a variable or does not
   *
   * \tparam T type of the node to which the method is attached. It must be given explicitly
   * if the method is inherited from a base class, otherwise the method will be registered
   * for this base class.
   */
  template<typename T, typename MethodT, typename EnumOutput, typename ConvertT = std::nullopt_t>
  void registerAccessor(EnumOutput o, MethodT method, ConvertT convert = std::nullopt);

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
  std::unique_ptr<ProbeNode> followUp(
      const Output & o,
      const std::function<bool(const Eigen::MatrixXd &)> & select = [](const Eigen::MatrixXd &) { return true; }) const;

  /** Follow up backward the computation graph starting at the inputs of \p g, and trimming out
   * branches where the value of an output does not pass the test given by \p select.
   */
  std::vector<std::unique_ptr<ProbeNode>> followUp(
      const graph::CallGraph * const g,
      const std::function<bool(const Eigen::MatrixXd &)> & select = [](const Eigen::MatrixXd &) { return true; }) const;

  /** Print in \p os the tree starting at \p root.*/
  void print(std::ostream & os, const std::unique_ptr<ProbeNode> & root) const;
  /** Print in \p os the trees starting each at an element of \p root.*/
  void print(std::ostream & os, const std::vector<std::unique_ptr<ProbeNode>> & roots) const;

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

  /** Add the value(s) corresponding to \p o to \p ov.*/
  bool addOutputVal(std::vector<OutputVal> & ov, const Output & o) const;
  /** Return the value(s) corresponding to \p o*/
  std::vector<OutputVal> outputVal(const Output & o) const;
  /** Subfunction for listOutputVal.*/
  std::vector<OutputVal> listOutputVal(const std::vector<graph::internal::Log::Output> & o, bool verbose) const;
  /** Subfunction for followUp*/
  std::unique_ptr<ProbeNode> followUp(const Output & o,
                                      const std::function<bool(const Eigen::MatrixXd &)> & select,
                                      Processed & processed) const;
  /** Subfunction for followUp*/
  std::unique_ptr<ProbeNode> followUp(const Update & u,
                                      const std::function<bool(const Eigen::MatrixXd &)> & select,
                                      Processed & processed) const;

  /** Subfunction for print*/
  void print(std::ostream & os, const std::unique_ptr<ProbeNode> & node, int depth) const;

  using OutputKey = std::pair<std::size_t, graph::internal::Log::EnumValue>;
  using VarMatrixPair = std::pair<VariablePtr, Eigen::MatrixXd>;
  const graph::internal::Log & log_;
  std::unordered_map<OutputKey, std::function<Eigen::MatrixXd(uintptr_t)>, internal::PairHasher> outputAccessor_;
  std::unordered_map<OutputKey, std::function<std::vector<VarMatrixPair>(uintptr_t)>, internal::PairHasher>
      varDepOutputAccessor_;
};

template<typename T, typename MethodT, typename EnumOutput, typename ConvertT>
inline void GraphProbe::registerAccessor(EnumOutput o, MethodT method, ConvertT convertIn)
{
  using CheckAccessor = internal::CheckAccessor<T, MethodT>;
  if constexpr(CheckAccessor::isVoidAccessor)
  {
    using ReturnT = typename CheckAccessor::ReturnT;
    auto convert = internal::MakeConvert<ReturnT>(std::move(convertIn));
    OutputKey k{std::type_index(typeid(T)).hash_code(), tvm::graph::internal::Log::EnumValue(o)};
    outputAccessor_[k] = [method, convert](uintptr_t t) { return convert((reinterpret_cast<T *>(t)->*method)()); };
  }
  else if constexpr(CheckAccessor::isVariableAccessor)
  {
    using ReturnT = typename CheckAccessor::ReturnT;
    auto convert = internal::MakeConvert<ReturnT>(std::move(convertIn));
    OutputKey k{std::type_index(typeid(T)).hash_code(), tvm::graph::internal::Log::EnumValue(o)};
    varDepOutputAccessor_[k] = [method, convert](uintptr_t t) {
      std::vector<VarMatrixPair> ret;
      T * ptr = reinterpret_cast<T *>(t);
      for(const auto & v : ptr->variables())
        ret.emplace_back(v, convert((ptr->*method)(*v)));
      return ret;
    };
  }
  else
  {
    static_assert(!std::is_same_v<T, T>, "Provided method does not have a valid signature");
  }
}

template<typename T>
inline void GraphProbe::registerTVMFunction()
{
  using GetVectorT = const Eigen::VectorXd & (T::*)() const;
  using GetJacobianT = tvm::internal::MatrixConstRefWithProperties (T::*)(const Variable &) const;
  using GetJDotT = MatrixConstRef (T::*)(const Variable &) const;
  registerAccessor<T>(T::Output::Value, static_cast<GetVectorT>(&T::value));
  registerAccessor<T>(T::Output::Velocity, static_cast<GetVectorT>(&T::velocity));
  registerAccessor<T>(T::Output::Jacobian, static_cast<GetJacobianT>(&T::jacobian));
  registerAccessor<T>(T::Output::NormalAcceleration, static_cast<GetVectorT>(&T::normalAcceleration));
  registerAccessor<T>(T::Output::JDot, static_cast<GetJDotT>(&T::JDot));
}

template<typename T>
inline void GraphProbe::registerTVMConstraint()
{
  using GetVectorT = const Eigen::VectorXd & (T::*)() const;
  using GetJacobianT = tvm::internal::MatrixConstRefWithProperties (T::*)(const Variable &) const;
  registerAccessor<T>(T::Output::Value, static_cast<GetVectorT>(&T::value));
  registerAccessor<T>(T::Output::Jacobian, static_cast<GetJacobianT>(&T::jacobian));
  registerAccessor<T>(T::Output::L, static_cast<GetVectorT>(&T::l));
  registerAccessor<T>(T::Output::U, static_cast<GetVectorT>(&T::u));
  registerAccessor<T>(T::Output::E, static_cast<GetVectorT>(&T::e));
}

template<typename T>
inline void GraphProbe::registerTVMTaskDynamics()
{
  using GetVectorT = const Eigen::VectorXd & (T::Impl::*)() const;
  registerAccessor<typename T::Impl>(T::Impl::Output::Value, static_cast<GetVectorT>(&T::Impl::value));
}
} // namespace tvm::diagnostic

TVM_DLLAPI std::ostream & operator<<(std::ostream & os,
                                     const std::vector<tvm::diagnostic::GraphProbe::OutputVal> & vals);
