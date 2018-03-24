#include <tvm/robot/Contact.h>

#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{


Contact::Contact(FramePtr f1, FramePtr f2,
                 std::vector<sva::PTransformd> points,
                 int ambiguityId)
: f1_(f1), f2_(f2), f1Points_(points),
  id_ { f1->robot().name(), f1->name(),
        f2->robot().name(), f2->name(),
        ambiguityId }

{
  // X_f1_f2 = X_0_f2 * X_0_f1.inv()
  X_f1_f2_ = f2_->position() * f1_->position().inv();
  X_f2_f1_ = X_f1_f2_.inv();
  for(const auto & X_f1_p : f1Points_)
  {
    f2Points_.emplace_back(X_f2_f1_ * X_f1_p);
  }
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

std::ostream & operator<<(std::ostream & os, const tvm::robot::Contact::Id & c)
{
  os << c.r1 << "::" << c.f1 << "/"
     << c.r2 << "::" << c.f2
     << " (id: " << c.ambiguityId << ")";
  return os;
}
