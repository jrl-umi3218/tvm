/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/robot/ConvexHull.h>

#include <tvm/utils/sch.h>

namespace tvm
{

namespace robot
{

ConvexHull::ConvexHull(const std::string & path, FramePtr f, const sva::PTransformd & X_f_o)
: ConvexHull(tvm::utils::Polyhedron(path), f, X_f_o)
{}

ConvexHull::ConvexHull(std::shared_ptr<sch::S_Object> o, FramePtr f, const sva::PTransformd & X_f_o)
: o_(o), f_(f), X_f_o_(X_f_o)
{
  registerUpdates(Update::Position, &ConvexHull::updatePosition);
  addOutputDependency(Output::Position, Update::Position);
  addInputDependency(Update::Position, f_, Frame::Output::Position);

  // Make sure the convex has the right position from the start
  updatePosition();
}

sch::CD_Pair ConvexHull::makePair(const ConvexHull & hull) const { return sch::CD_Pair(o_.get(), hull.o_.get()); }

void ConvexHull::updatePosition() { tvm::utils::transform(*o_, X_f_o_ * f_->position()); }

const Frame & ConvexHull::frame() const { return *f_; }

Frame & ConvexHull::frame() { return *f_; }

} // namespace robot

} // namespace tvm
