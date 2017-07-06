#include <memory>

#include <tvm/api.h>
#include "ConstraintEnums.h"

namespace tvm
{
  class Function;
  class TaskDynamics;

  /** A conveniency proxy to represents expression f==0, f>=0 or f<=0 where f
    * is a function
    */
  struct TVM_DLLAPI ProtoTask
  {
    std::shared_ptr<Function> f_;
    ConstraintType type_;
  };

  /** For now, we only accept rhs=0*/
  ProtoTask operator==(std::shared_ptr<Function> f, double rhs);
  ProtoTask operator>=(std::shared_ptr<Function> f, double rhs);
  ProtoTask operator<=(std::shared_ptr<Function> f, double rhs);


  /** A task is a triplet (Function, operator, TaskDynamics) where operator is
    * ==, >= or <=*/
  class TVM_DLLAPI Task
  {
  public:
    Task(std::shared_ptr<Function> f, ConstraintType t, std::shared_ptr<TaskDynamics> td);
    Task(ProtoTask proto, std::shared_ptr<TaskDynamics> td);

    std::shared_ptr<Function> function() const;
    ConstraintType type() const;
    std::shared_ptr<TaskDynamics> taskDynamics() const;

  private:
    std::shared_ptr<Function> f_;
    ConstraintType type_;
    std::shared_ptr<TaskDynamics> td_;
  };
}