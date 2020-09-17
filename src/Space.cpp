/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Space.h>

#include <tvm/Variable.h>

#include <sstream>

namespace tvm
{
Space::Space(int size) : Space(size, size, size) {}

Space::Space(int size, int representationSize) : Space(size, representationSize, size) {}

Space::Space(int size, int representationSize, int tangentRepresentationSize)
: mSize_(size), rSize_(representationSize), tSize_(tangentRepresentationSize),
  type_((size == representationSize && size == tangentRepresentationSize) ? Type::Euclidean : Type::Unspecified)
{
  assert(size >= 0);
  assert(representationSize >= size);
  assert(tangentRepresentationSize >= size);
}

Space::Space(Type type, int size) : type_(type)
{
  switch(type)
  {
    case Type::Euclidean:
      mSize_ = rSize_ = tSize_ = size;
      break;
    case Type::SO3:
      assert(size < 0);
      mSize_ = 3;
      rSize_ = 4;
      tSize_ = 3;
      break;
    case Type::SE3:
      assert(size < 0);
      mSize_ = 6;
      rSize_ = 7;
      tSize_ = 6;
      break;
    case Type::Unspecified:
    default:
      throw std::runtime_error("[Space::Space] Unable to build the required space.");
  }
}

std::unique_ptr<Variable> Space::createVariable(std::string_view name) const
{
  return std::unique_ptr<Variable>(new Variable(*this, name));
}

int Space::size() const { return mSize_; }

int Space::rSize() const { return rSize_; }

int Space::tSize() const { return tSize_; }

Space::Type Space::type() const { return type_; }

bool Space::isEuclidean() const { return type_ == Type::Euclidean; }

} // namespace tvm
