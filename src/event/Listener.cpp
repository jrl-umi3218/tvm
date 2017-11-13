#include <tvm/event/Listener.h>

namespace tvm
{

namespace event
{

  void Listener::receive(Type evt, const Source & notifier)
  {
    process(evt, notifier);
  }

}

}  // namespace tvm
