#include <tvm/robot/Contact.h>

#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{


Contact::Contact(FramePtr f1, FramePtr f2,
                 std::vector<sva::PTransformd> points,
                 int ambiguityId)
: f1_(f1), f2_(f2), points_(points),
  id_ { f1->robot().name(), f1->name(),
        f2->robot().name(), f2->name(),
        ambiguityId }

{
  addDirectDependency(Output::F1Position, f1, Frame::Output::Position);
  addDirectDependency(Output::F1Jacobian, f1, Frame::Output::Jacobian);
  addDirectDependency(Output::F1Velocity, f1, Frame::Output::Velocity);
  addDirectDependency(Output::F1NormalAcceleration, f1, Frame::Output::NormalAcceleration);
  addDirectDependency(Output::F2Position, f2, Frame::Output::Position);
  addDirectDependency(Output::F2Jacobian, f2, Frame::Output::Jacobian);
  addDirectDependency(Output::F2Velocity, f2, Frame::Output::Velocity);
  addDirectDependency(Output::F2NormalAcceleration, f2, Frame::Output::NormalAcceleration);
}

}

}
