/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <tvm/api.h>
#include <tvm/event/enums.h>

#include <map>
#include <memory>
#include <vector>

namespace tvm
{

namespace event
{

class Listener;

/* todo or to consider:
  - delayed notify in case the same notification might be issue several time in a row
    (for example, several contacts are added/remove at several point in the code)
  - time-stamped events or lazy processing, so that a same event arriving by two
    different way to a listener is not processed twice.*/

class TVM_DLLAPI Source
{
public:
  virtual ~Source() = default;

  void regist(std::shared_ptr<Listener> user, Type evt);
  void notify(Type evt);

private:
  /** internal we use weak_ptr here to avoid creating cyclic dependencies when a class
   * inheriting from both EventSource and DataSource refers to and is refered by a
   * class inheriting from DataNode and EventListener.
   */
  std::map<Type, std::vector<std::weak_ptr<Listener>>> registrations_;
};

} // namespace event

} // namespace tvm
