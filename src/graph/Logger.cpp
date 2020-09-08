/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/internal/Logger.h>

#include <tvm/graph/CallGraph.h>

namespace tvm
{

namespace graph
{

namespace internal
{

Logger & Logger::logger()
{
  static std::unique_ptr<Logger> logger_{new Logger()};
  return *logger_;
}

void Logger::disable() { disabled_ = true; }

void Logger::enable() { disabled_ = false; }

void Logger::addGraphOutput(CallGraph * g, Inputs * node)
{
  if(disabled_)
    return;

  log_.graphOutputs_[g].push_back(node);
}

} // namespace internal

} // namespace graph

} // namespace tvm
