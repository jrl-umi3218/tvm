#include <tvm/graph/internal/Inputs.h>

namespace tvm
{

namespace graph
{

namespace internal
{

Inputs::Iterator::Iterator(inputs_t::iterator it, inputs_t::iterator end)
: Inputs::inputs_t::iterator(it),
  end_(end)
{
}

Inputs::Iterator::operator bool()
{
  return *this != end_;
}

}  // namespace internal

}  // namespace graph

}  // namespace tvm
