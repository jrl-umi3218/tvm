/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/event/Source.h>

#include <tvm/event/Listener.h>

namespace tvm
{

namespace event
{

void Source::regist(std::shared_ptr<Listener> user, Type evt) { registrations_[evt].push_back(user); }

void Source::notify(Type evt)
{
  const auto & list = registrations_[evt];
  for(const auto & u : list)
    u.lock()->receive(evt, *this);
}

} // namespace event

} // namespace tvm
