#pragma once

#include <map>
#include <memory>
#include <vector>

namespace taskvm
{
  class EventListener;

  enum DataEvent
  {
    CONTACT_NUMBER_CHANGED,
  };

  /* todo or to consider: 
    - delayed notify in case the same notification might be issue several time in a row
      (for example, several contacts are added/remove at several point in the code)
    - time-stamped events or lazy processing, so that a same event arriving by two 
      different way to a listener is not processed twice.*/

  class EventSource
  {
  public:
    virtual ~EventSource() = default;

    void registerUser(std::shared_ptr<EventListener> user, DataEvent evt);
    void notify(DataEvent evt);

  private:
    /** internal we use weak_ptr here to avoid creating cyclic dependencies when a class
      * inheriting from both EventSource and DataSource refers to and is refered by a
      * class inheriting from DataNode and EventListener.
      */
    std::map<DataEvent, std::vector<std::weak_ptr<EventListener>>> eventRegistrations_;
  };

  class EventListener
  {
  public:
    virtual ~EventListener() = default;

    void receive(DataEvent evt, const EventSource& notifier);  //TODO: should it be const? -> would surely require mutable in derived classes

  protected:
    virtual void process(DataEvent evt, const EventSource& notifier) = 0;
  };
}