/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/robot/utils.h>

#include <tvm/exception/exceptions.h>

#include <RBDyn/parsers/urdf.h>

#include <iostream>

namespace tvm
{

namespace robot
{

std::unique_ptr<Robot> fromURDF(tvm::Clock & clock,
                                const std::string & name,
                                const std::string & path,
                                bool fixed,
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
  auto data = rbd::parsers::from_urdf(ss.str(), fixed, filteredLinks);
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
      std::cerr << "Joint " << qi.first << ": provided configuration has " << qi.second.size()
                << " params but loaded robot has " << init_q[jIndex].size() << " params." << std::endl;
      continue;
    }
    init_q[jIndex] = qi.second;
  }
  data.mbc.q = init_q;
  return std::unique_ptr<Robot>{new Robot(clock, name, data.mbg, data.mb, data.mbc, data.limits)};
}

} // namespace robot

} // namespace tvm
