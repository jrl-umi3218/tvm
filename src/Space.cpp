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
  {
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

  bool Space::isEuclidean() const
  {
    return mSize_ == rSize_;
  }

}  // namespace tvm
