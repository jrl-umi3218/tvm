/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include "RobotPublisher.h"

struct RobotPublisherImpl
{
};

RobotPublisher::RobotPublisher(const std::string &) : impl_() {}

RobotPublisher::~RobotPublisher() {}

void RobotPublisher::publish(const tvm::Robot &) {}
