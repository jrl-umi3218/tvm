#include "RobotPublisher.h"

struct RobotPublisherImpl {};

RobotPublisher::RobotPublisher(const std::string &):
  impl_()
{}

RobotPublisher::~RobotPublisher() {}

void RobotPublisher::publish(const tvm::Robot &) {}
