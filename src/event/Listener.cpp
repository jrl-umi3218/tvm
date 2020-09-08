/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/event/Listener.h>

namespace tvm
{

namespace event
{

void Listener::receive(Type evt, const Source & notifier) { process(evt, notifier); }

} // namespace event

} // namespace tvm
