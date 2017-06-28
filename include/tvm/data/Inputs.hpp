#pragma once

#include <algorithm>
#include <sstream>

namespace
{
  template<typename T>
  inline void check_output_enabled(const std::shared_ptr<T> &) {}

  template<typename T, typename EnumT, typename ... Args>
  inline void check_output_enabled(const std::shared_ptr<T> & s, EnumT o, Args ... args)
  {
    if(!s->isOutputEnabled(static_cast<int>(o)))
    {
      std::stringstream ss;
      ss << "Output " << T::OutputName(o) << " is not enabled in " << T::OutputBaseName << " (or derived)";
      throw std::runtime_error(ss.str());
    }
    if(sizeof...(Args))
    {
      check_output_enabled(s, args...);
    }
  }
}

namespace tvm
{

namespace data
{

template<typename T, typename ... Args>
void Inputs::addInput(std::shared_ptr<T> source, Args ... args)
{
  static_assert(is_valid_output<T>(Args()...), "One of the outputs you requested is not part of the provided source");
  check_output_enabled(source, args...);
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
