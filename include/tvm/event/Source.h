/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>
#include <tvm/event/enums.h>

#include <map>
#include <memory>
#include <vector>

namespace tvm
{

namespace event
{

class Listener;

/* todo or to consider:
  - delayed notify in case the same notification might be issue several time in a row
    (for example, several contacts are added/remove at several point in the code)
  - time-stamped events or lazy processing, so that a same event arriving by two
    different way to a listener is not processed twice.*/

class TVM_DLLAPI Source
{
public:
  virtual ~Source() = default;

  void regist(std::shared_ptr<Listener> user, Type evt);
  void notify(Type evt);

private:
  /** internal we use weak_ptr here to avoid creating cyclic dependencies when a class
   * inheriting from both EventSource and DataSource refers to and is refered by a
   * class inheriting from DataNode and EventListener.
   */
  std::map<Type, std::vector<std::weak_ptr<Listener>>> registrations_;
};

} // namespace event

} // namespace tvm
