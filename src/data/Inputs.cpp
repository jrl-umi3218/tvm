#include "tvm/data/Inputs.h"

namespace tvm
{

namespace data
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

}

}
