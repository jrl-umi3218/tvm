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

#include <tvm/Space.h>

#include <tvm/Variable.h>

#include <sstream>

namespace tvm
{
  Space::Space(int size)
    : Space(size, size, size)
  {
  }

  Space::Space(int size, int representationSize)
    : Space(size, representationSize, size)
  {
  }

  Space::Space(int size, int representationSize, int tangentRepresentationSize)
    : mSize_(size), rSize_(representationSize), tSize_(tangentRepresentationSize)
    , type_((size==representationSize&&size==tangentRepresentationSize)?Type::Euclidean:Type::Unspecified)
  {
    assert(size >= 0);
    assert(representationSize >= size);
    assert(tangentRepresentationSize >= size);
  }

  Space::Space(Type type, int size)
    : type_(type)
  {
    switch (type)
    {
    case Type::Euclidean: mSize_ = rSize_ = tSize_ = size; break;
    case Type::SO3: assert(size < 0); mSize_ = 3; rSize_ = 4; tSize_ = 3; break;
    case Type::SE3: assert(size < 0); mSize_ = 6; rSize_ = 7; tSize_ = 6; break;
    case Type::Unspecified:
    default:
      throw std::runtime_error("[Space::Space] Unable to build the required space.");
    }
  }

  std::unique_ptr<Variable> Space::createVariable(const std::string& name) const
  {
    return std::unique_ptr<Variable>(new Variable(*this, name));
  }

  int Space::size() const
  {
    return mSize_;
  }

  int Space::rSize() const
  {
    return rSize_;
  }

  int Space::tSize() const
  {
    return tSize_;
  }

  Space::Type Space::type() const
  {
    return type_;
  }

  bool Space::isEuclidean() const
  {
    return type_ == Type::Euclidean;
  }

}  // namespace tvm
