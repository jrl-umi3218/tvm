#include "exceptions.h"
#include "Function.h"
#include "Variable.h"

namespace tvm
{
  Function::Function(int m)
    : FirstOrderProvider(m)
  {
    resizeVelocityCache();
    resizeNormalAccelerationCache();
    resizeJDotCache();
  }

  void Function::resizeCache()
  {
    FirstOrderProvider::resizeCache();
    resizeVelocityCache();
    resizeNormalAccelerationCache();
    resizeJDotCache();
  }

  void Function::resizeVelocityCache()
  {
    if (isOutputEnabled((int)Output::Velocity))
      velocity_.resize(size());
  }

  void Function::resizeNormalAccelerationCache()
  {
    if (isOutputEnabled((int)Output::NormalAcceleration))
      normalAcceleration_.resize(size());
  }

  void Function::resizeJDotCache()
  {
    if (isOutputEnabled((int)Output::JDot))
    {
      for (auto v : variables())
        JDot_[v.get()].resize(size(), v->space().tSize());
    }
  }

  void Function::addVariable_(VariablePtr v)
  {
    JDot_[v.get()].resize(size(), v->space().tSize());
    variablesDot_.push_back(dot(v));
  }

  void Function::removeVariable_(VariablePtr v)
  {
    JDot_.erase(v.get());
    auto it = std::find(variablesDot_.begin(), variablesDot_.end(), dot(v));
    assert(it != variablesDot_.end()
      && "This should not happen: FirstOrderProvider::removeVariable would raise an exception first if the variable was not there.");
    variablesDot_.erase(it);
  }
}
