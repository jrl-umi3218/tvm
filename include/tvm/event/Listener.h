#pragma once

#include <tvm/event/Source.h>

#include <map>
#include <memory>
#include <vector>

namespace tvm
{

namespace event
{

  class TVM_DLLAPI Listener
  {
  public:
    virtual ~Listener() = default;

    void receive(Type evt, const Source& notifier);  //TODO: should it be const? -> would surely require mutable in derived classes

  protected:
    virtual void process(Type evt, const Source& notifier) = 0;
  };

}  // namespace event

}  // namespace tvm
