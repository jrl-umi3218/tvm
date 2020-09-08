/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/Robot.h>

struct RobotPublisherImpl;

class RobotPublisher
{
public:
  RobotPublisher(const std::string & prefix);

  ~RobotPublisher();

  void publish(const tvm::Robot & robot);

private:
  std::unique_ptr<RobotPublisherImpl> impl_;
};
