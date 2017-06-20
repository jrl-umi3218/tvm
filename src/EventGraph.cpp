#include "EventGraph.h"

namespace taskvm
{
  void EventSource::registerUser(std::shared_ptr<EventListener> user, DataEvent evt)
  {
    eventRegistrations_[evt].push_back(user);
  }

  void EventSource::notify(DataEvent evt)
  {
    const auto list = eventRegistrations_[evt];
    for (const auto u : list)
      u.lock()->receive(evt, *this);
  }

  void EventListener::receive(DataEvent evt, const EventSource & notifier)
  {
    process(evt, notifier);
  }

}
