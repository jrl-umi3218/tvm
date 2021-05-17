/** Copyright 2017-2021 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Variable.h>
#include <tvm/api.h>

#include <tvm/diagnostic/internal/probe.h>
#include <tvm/graph/CallGraph.h>

#include <Eigen/Core>

#include <iosfwd>
#include <map>
#include <tuple>
#include <vector>

namespace tvm::diagnostic
{
class TVM_DLLAPI GraphProbe
{
public:
  using OutputVal = std::tuple<graph::internal::Log::Output, VariablePtr, Eigen::MatrixXd>;

  GraphProbe(const graph::internal::Log & log);

  template<typename T, typename U, typename EnumOutput>
  void registerAccessor(EnumOutput o, const U & (T::*fn)() const);

  template<typename T, typename U, typename EnumOutput>
  void registerAccessor(EnumOutput o, U (T::*fn)(const Variable &) const);

  std::vector<OutputVal> listOutputVal(bool verbose = false) const;

private:
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
} // namespace tvm::diagnostic