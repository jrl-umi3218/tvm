#include <memory>

#include <tvm/api.h>
#include "ConstraintEnums.h"
#include "defs.h"

namespace tvm
{
  /** A conveniency proxy to represents expression f==0, f>=0 or f<=0 where f
    * is a function
    */
  class TVM_DLLAPI ProtoTask
  {
  public:
    FunctionPtr f_;
    ConstraintType type_;
  };

  /** For now, we only accept rhs=0*/
  ProtoTask operator==(FunctionPtr f, double rhs);
  ProtoTask operator>=(FunctionPtr f, double rhs);
  ProtoTask operator<=(FunctionPtr f, double rhs);


  /** A task is a triplet (Function, operator, TaskDynamics) where operator is
    * ==, >= or <=*/
  class TVM_DLLAPI Task
  {
  public:
    Task(FunctionPtr f, ConstraintType t, std::shared_ptr<TaskDynamics> td);
    Task(ProtoTask proto, std::shared_ptr<TaskDynamics> td);

    FunctionPtr function() const;
    ConstraintType type() const;
    std::shared_ptr<TaskDynamics> taskDynamics() const;

  private:
    FunctionPtr f_;
    ConstraintType type_;
    std::shared_ptr<TaskDynamics> td_;
  };
}