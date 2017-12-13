#include <tvm/graph/internal/Logger.h>

namespace tvm
{

namespace graph
{

namespace internal
{

  Logger& Logger::logger()
  {
    static std::unique_ptr<Logger> logger_{ new Logger() };
    return *logger_;
  }

  void Logger::addGraphOutput(CallGraph* g, Inputs* node)
  {
    log_.graphOutputs_[g].push_back(node);
  }


} // namespace internal

} // namespace graph

} // namespace tvm
