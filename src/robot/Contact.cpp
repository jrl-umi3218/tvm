/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <tvm/robot/Contact.h>

#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{

Contact::Contact(FramePtr f1, FramePtr f2, std::vector<sva::PTransformd> points, int ambiguityId)
: f1_(f1), f2_(f2), f1Points_(points), id_{f1->robot().name(), f1->name(), f2->robot().name(), f2->name(), ambiguityId}

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

} // namespace robot

} // namespace tvm

std::ostream & operator<<(std::ostream & os, const tvm::robot::Contact::Id & c)
{
  os << c.r1 << "::" << c.f1 << "/" << c.r2 << "::" << c.f2 << " (id: " << c.ambiguityId << ")";
  return os;
}
