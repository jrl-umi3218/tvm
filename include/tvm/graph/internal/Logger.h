#pragma once

#include <tvm/api.h>
#include <tvm/graph/internal/Log.h>

#include <algorithm>
#include <fstream>
#include <memory>

namespace tvm
{
  
namespace graph
{

namespace internal
{
  class Inputs;

  class TVM_DLLAPI Logger
  {
  public:
    static Logger& logger();

    const Log& log() const;

    template<typename U, typename EnumT>
    void registerUpdate(U* node, EnumT u, void(U::*fn)());

    template<typename S, typename EnumO>
    void addInput(Inputs* node, S* source, EnumO i);

    template<typename U, typename EnumO, typename EnumU>
    void addOutputDependency(U* node, EnumO o, EnumU u);

    template<typename U, typename EnumU1, typename EnumU2>
    void addInternalDependency(U* node, EnumU1 uDependent, EnumU2 u);

    template<typename U, typename EnumU, typename S, typename EnumO>
    void addInputDependency(U* node, EnumU u, S* source, EnumO i);

    template<typename U, typename EnumO, typename S, typename EnumI>
    void addDirectDependency(U* node, EnumO o, S* source, EnumI i);

    /** Register the type associated to a pointer. */
    template<typename U>
    void registerType(U* node);

    template<typename U>
    void logCall(U* node, void(U::*fn)());

  private:
    Logger() = default;

    //singleton
    static std::unique_ptr<Logger> logger_;
    
    //raw log
    Log log_;

    //output file
    std::ofstream aof;
  };

  //Helper function for pointer-to-member-function
  template <typename T>
  std::uintptr_t getPointerValue(void (T::*ptfm)())
  {
    auto cptr = reinterpret_cast<std::uintptr_t*>(&ptfm);
    return *cptr;
  }

  inline const Log& Logger::log() const
  {
    return log_;
  }

  template<typename U, typename EnumT>
  inline void Logger::registerUpdate(U* node, EnumT u, void(U::* fn)())
  {
    Log::Update up{Log::EnumValue(u),
                   U::UpdateName(u),
                   getPointerValue<U>(fn),
                   Log::Pointer(node) };
    log_.updates_.push_back(up);
    registerType(node);
  }

  template<typename S, typename EnumO>
  inline void Logger::addInput(Inputs* node, S* source, EnumO i)
  {
    Log::Input in = {Log::EnumValue(i),
                     S::OutputName(i),
                     Log::Pointer(source),
                     Log::Pointer(node)};
    log_.inputs_.push_back(in);
    registerType(node);
    registerType(source);
  }

  template<typename U, typename EnumO, typename EnumU>
  inline void Logger::addOutputDependency(U* node, EnumO o, EnumU u)
  {
    Log::Output out = {Log::EnumValue(o),
                       U::OutputName(o),
                       Log::Pointer(node)};
    log_.outputs_.push_back(out);

    Log::OutputDependency dep = {Log::EnumValue(u), 
                                 Log::EnumValue(o),
                                 Log::Pointer(node)};
    log_.outputDependencies_.push_back(dep);
    registerType(node);
  }


  template<typename U, typename EnumU1, typename EnumU2>
  inline void Logger::addInternalDependency(U* node, EnumU1 uDependent, EnumU2 u)
  {
    Log::InternalDependency dep = {Log::EnumValue(u), 
                                   Log::EnumValue(uDependent),
                                   Log::Pointer(node)};
    log_.internalDependencies_.push_back(dep);
    registerType(node);
  }

  template<typename U, typename EnumU, typename S, typename EnumO>
  inline void Logger::addInputDependency(U* node, EnumU u, S* source, EnumO i)
  {
    Log::InputDependency dep = {Log::EnumValue(i),
                                Log::EnumValue(u),
                                Log::Pointer(source),
                                Log::Pointer(node)};
    log_.inputDependencies_.push_back(dep);
    registerType(node);
    registerType(source);
  }

  template<typename U, typename EnumO, typename S, typename EnumI>
  inline void Logger::addDirectDependency(U* node, EnumO o, S* source, EnumI i)
  {
    Log::DirectDependency dep = {Log::EnumValue(i),
                                 Log::EnumValue(o),
                                 Log::Pointer(source),
                                 Log::Pointer(node)};
    log_.directDependencies_.push_back(dep);
    registerType(node);
    registerType(source);
  }

  template<typename U>
  inline void Logger::registerType(U* node)
  {
    Log::TypeInfo t(typeid(*node));
    std::uintptr_t val = reinterpret_cast<std::uintptr_t>(node);
    auto& types = log_.types_[val];
    auto it = std::find(types.begin(), types.end(), t);
    if (it == types.end())
    {
      types.push_back(t);
    }
  }

#define TVM_GRAPH_LOG_REGISTER_UPDATE(node, u, fn) \
  tvm::graph::internal::Logger::logger().registerUpdate<U,EnumT>(static_cast<U*>(node), u, fn);

#define TVM_GRAPH_LOG_ADD_INPUT(add, node, i, source) \
  if (add) {tvm::graph::internal::Logger::logger().addInput<T,EnumI>(node,source.get(),i);}

#define TVM_GRAPH_LOG_ADD_OUTPUT_DEPENDENCY(node, o, u) \
  tvm::graph::internal::Logger::logger().addOutputDependency<U,EnumO,EnumU>(static_cast<U*>(node), o, u);

#define TVM_GRAPH_LOG_ADD_INTERNAL_DEPENDENCY(node, uDependent, u) \
  tvm::graph::internal::Logger::logger().addInternalDependency<U,EnumU1,EnumU2>(static_cast<U*>(node), uDependent, u);

#define TVM_GRAPH_LOG_ADD_INPUT_DEPENDENCY(node, u, source, i) \
  tvm::graph::internal::Logger::logger().addInputDependency<U,EnumU,S,EnumO>(static_cast<U*>(node), u, source.get(), i);

#define TVM_GRAPH_LOG_ADD_DIRECT_DEPENDENCY(node, o, source, i) \
  tvm::graph::internal::Logger::logger().addDirectDependency(static_cast<U*>(node), o, source.get(), i);

} // namespace internal

}  //namespace graph

} // namespace tvm
