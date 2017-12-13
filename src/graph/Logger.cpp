#include <tvm/graph/internal/Logger.h>

namespace tvm
{

namespace graph
{

namespace internal
{

  std::unique_ptr<Logger> Logger::logger_;



  Logger& Logger::logger()
  {
    if (!logger_)
    {
      logger_.reset(new Logger());
    }
    return *logger_;
  }

  void Logger::addGraphOutput(CallGraph* g, Inputs* node)
  {
    log_.graphOutputs_[g].push_back(node);
  }


} // namespace internal

} // namespace graph

} // namespace tvm
