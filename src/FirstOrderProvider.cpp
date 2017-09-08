#include "exceptions.h"
#include "FirstOrderProvider.h"
#include "Variable.h"

namespace tvm
{
  namespace internal
  {
    FirstOrderProvider::FirstOrderProvider(int m)
      : m_(m)
    {
      resizeCache(); //resize value_
    }
    
    void FirstOrderProvider::resizeCache()
    {
      resizeValueCache();
      resizeJacobianCache();
    }

    void FirstOrderProvider::resizeValueCache()
    {
      if (isOutputEnabled((int)Output::Value))
        value_.resize(m_);
    }

    void FirstOrderProvider::resizeJacobianCache()
    {
      if (isOutputEnabled((int)Output::Jacobian))
      {
        for (auto v : variables_)
          jacobian_[v.get()].resize(m_, v->size());
      }
    }

    void FirstOrderProvider::addVariable(VariablePtr v)
    {
      if (std::find(variables_.begin(), variables_.end(), v) == variables_.end())
        variables_.push_back(v);
      else
        throw DuplicateVariable(/*desc*/); //TODO

      jacobian_[v.get()].resize(m_, v->size());

      addVariable_(v);
    }

    void FirstOrderProvider::removeVariable(VariablePtr v)
    {
      auto it = std::find(variables_.begin(), variables_.end(), v);
      if (it == variables_.end())
        throw NonExistingVariable(/*desc*/); //TODO
      else
      {
        variables_.erase(it);
        jacobian_.erase(v.get());
      }

      removeVariable_(v);
    }

    void FirstOrderProvider::addVariable_(VariablePtr)
    {
      //do nothing
    }

    void FirstOrderProvider::removeVariable_(VariablePtr)
    {
      //do nothing
    }
  }
}