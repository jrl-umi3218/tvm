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

#include <tvm/utils/sch.h>

namespace tvm
{

namespace utils
{

  void transform(sch::S_Object& obj, const sva::PTransformd& t)
  {
    sch::Matrix4x4 m;
    const Eigen::Matrix3d& rot = t.rotation();
    const Eigen::Vector3d& tran = t.translation();

    for(unsigned int i = 0; i < 3; ++i)
    {
      for(unsigned int j = 0; j < 3; ++j)
      {
        m(i,j) = rot(j,i);
      }
    }

    m(0,3) = tran(0);
    m(1,3) = tran(1);
    m(2,3) = tran(2);

    obj.setTransformation(m);
  }

  std::unique_ptr<sch::S_Polyhedron> Polyhedron(const std::string& filename)
  {
    auto s = std::unique_ptr<sch::S_Polyhedron>(new sch::S_Polyhedron{});
    s->constructFromFile(filename);
    return s;
  }


  double distance(sch::CD_Pair& pair, Eigen::Vector3d& p1, Eigen::Vector3d& p2)
  {
    sch::Point3 p1Tmp, p2Tmp;
    double dist = pair.getClosestPoints(p1Tmp, p2Tmp);

    p1 << p1Tmp[0], p1Tmp[1], p1Tmp[2];
    p2 << p2Tmp[0], p2Tmp[1], p2Tmp[2];

    return dist;
  }

} // utils

} // namespace tvm
