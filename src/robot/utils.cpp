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

#include <tvm/robot/utils.h>

#include <tvm/exception/exceptions.h>

#include <mc_rbdyn_urdf/urdf.h>

#include <iostream>

namespace tvm
{

namespace robot
{

std::unique_ptr<Robot> fromURDF(tvm::Clock & clock, const std::string & name,
                                const std::string & path, bool fixed,
                                const std::vector<std::string> & filteredLinks,
                                const std::map<std::string, std::vector<double>> & q)
{
  std::ifstream ifs(path);
  if(!ifs.good())
  {
    throw tvm::exception::DataException("Failed to open " + path);
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  auto data = mc_rbdyn_urdf::rbdyn_from_urdf(ss.str(), fixed, filteredLinks);
  data.mbc.gravity = tvm::constant::gravity;
  auto init_q = data.mbc.q;
  const auto & jIndexByName = data.mb.jointIndexByName();
  for(const auto & qi : q)
  {
    if(!jIndexByName.count(qi.first))
    {
      continue;
    }
    auto jIndex = jIndexByName.at(qi.first);
    if(init_q[jIndex].size() != qi.second.size())
    {
      std::cerr << "Joint " << qi.first << ": provided configuration has " << qi.second.size() << " params but loaded robot has " << init_q[jIndex].size() << " params." << std::endl;
      continue;
    }
    init_q[jIndex] = qi.second;
  }
  data.mbc.q = init_q;
  return std::unique_ptr<Robot>{new Robot(clock, name, data.mbg, data.mb, data.mbc, data.limits)};
}

} // namespace robot

} // namespace tvm
