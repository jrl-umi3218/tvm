/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/internal/PairElementToken.h>

#include <functional>
#include <vector>

namespace tvm::internal
{
/** A class to register and run callbacks, based on PairElementToken to identify
 * the class who registered the callback and check if it is still alive.
 * If this class was destroyed, the callback is automatically unregistered.
 */
class CallbackManager
{
public:
  using Callback = std::function<void()>;
  using TCPair = std::pair<PairElementToken, Callback>;

  CallbackManager() = default;
  CallbackManager(const CallbackManager &) = delete;
  CallbackManager & operator=(const CallbackManager &) = delete;

  virtual ~CallbackManager() = default;

  /** Register a callback and return a handle that needs to be converted to
   * PairElementToken.
   */
  PairElementTokenHandle registerCallback(std::function<void()> c);

  /** Unregister a callback, identified by the token created from its registration.*/
  void unregisterCallback(const PairElementToken & t);

  /** Run all callbacks. Remove the ones for which the token obtained at the
   * registration was destroyed.
   */
  void run();

private:
  std::vector<TCPair> callbacks_;
};

inline PairElementTokenHandle CallbackManager::registerCallback(std::function<void()> c)
{
  callbacks_.emplace_back(PairElementToken(), c);
  return PairElementTokenHandle(callbacks_.back().first);
}

inline void CallbackManager::unregisterCallback(const PairElementToken & t)
{
  auto it =
      std::find_if(callbacks_.begin(), callbacks_.end(), [&](const TCPair & p) { return p.first.isPairedWith(t); });
  if(it != callbacks_.end())
    callbacks_.erase(it);
  else
    throw std::runtime_error("Element to unregister not found.");
}

inline void CallbackManager::run()
{
  for(size_t i = 0; i < callbacks_.size();)
  {
    if(callbacks_[i].first.isPaired())
    {
      std::invoke(callbacks_[i].second);
      ++i;
    }
    else
    {
      callbacks_.erase(callbacks_.begin() + i);
    }
  }
}

} // namespace tvm::internal
