/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/internal/Inputs.h>

namespace tvm
{

namespace graph
{

namespace internal
{

Inputs::Iterator::Iterator(inputs_t::iterator it, inputs_t::iterator end) : Inputs::inputs_t::iterator(it), end_(end) {}

Inputs::Iterator::operator bool() { return *this != end_; }

void Inputs::removeInput(Iterator it, abstract::Outputs * source)
{
  inputs_.erase(it);
  auto sourceIt = std::find_if(store_.begin(), store_.end(),
                               [source](const std::shared_ptr<abstract::Outputs> & o) { return o.get() == source; });
  if(sourceIt != store_.end())
  {
    store_.erase(sourceIt);
  }
}

} // namespace internal

} // namespace graph

} // namespace tvm
