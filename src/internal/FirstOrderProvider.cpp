#include <tvm/internal/FirstOrderProvider.h>

#include <tvm/Variable.h>
#include <tvm/exception/exceptions.h>

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
      for (auto v : variables_.variables())
        jacobian_[v.get()].resize(m_, v->space().tSize());
    }
  }

  void FirstOrderProvider::addVariable(VariablePtr v, bool linear)
  {
    variables_.add(v);

    jacobian_[v.get()].resize(m_, v->space().tSize());
    linear_[v.get()] = linear;

    addVariable_(v);
  }

  void FirstOrderProvider::addVariable(const VariableVector & vv, bool linear)
  {
    for(auto v : vv.variables())
    {
      addVariable(v, linear);
    }
  }

  void FirstOrderProvider::removeVariable(VariablePtr v)
  {
    variables_.remove(*v);
    jacobian_.erase(v.get());
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

  void FirstOrderProvider::splitJacobian(const MatrixConstRef & J, const std::vector<VariablePtr>& vars, bool keepProperties)
  {
    Eigen::DenseIndex s = 0;
    for (const auto& v : vars)
    {
      auto n = static_cast<Eigen::DenseIndex>(v->space().tSize());
      jacobian_[v.get()].keepProperties(keepProperties) = J.middleCols(s, n);
      s += n;
    }
  }

}  // namespace internal

}  // namespace tvm
