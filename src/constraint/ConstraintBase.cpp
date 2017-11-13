#include <tvm/constraint/internal/ConstraintBase.h>

namespace tvm
{

namespace constraint
{

namespace internal
{

ConstraintBase::ConstraintBase(int m)
  : FirstOrderProvider(m)
{
}

}

}  // namespace constraint

}  // namespace tvm
