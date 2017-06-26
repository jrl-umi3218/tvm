#pragma once

#include <algorithm>

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

template<typename T, typename ... Args>
void Inputs::addInput(std::shared_ptr<T> source, const Args ... args)
{
  static_assert(is_valid_output<T>(Args()...), "One of the outputs you requested is not part of the provided source");
  auto it = getInput(source);
  std::set<int> v { static_cast<int>(args)... };
  if(!it)
  {
    inputs_[source] = std::move(v);
  }
  else
  {
    for(auto i : v)
    {
      it->second.insert(i);
    }
  }
}

template<typename T>
Inputs::Iterator Inputs::getInput(std::shared_ptr<T> source)
{
  static_assert(std::is_base_of<Outputs, T>::value, "Inputs cannot store outputs that do not derived from Ouputs.");
  // FIXME Use something more clever or a better data structure to search for inputs
  return { std::find_if(inputs_.begin(), inputs_.end(),
                        [&source](const typename inputs_t::value_type & p)
                        {
                          return p.first.get() == source.get();
                        }),
            inputs_.end()};
}

} // data

} // tvm
