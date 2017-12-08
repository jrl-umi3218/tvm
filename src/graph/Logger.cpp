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

} // namespace internal

} // namespace graph

} // namespace tvm
