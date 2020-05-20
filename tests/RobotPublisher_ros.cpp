/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include "RobotPublisher.h"

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <tf2_ros/transform_broadcaster.h>


namespace
{

bool ros_init()
{
  int argc = 0;
  char * argv[] = {0};
  ros::init(argc, argv, "tvm_test");
  if(!ros::master::check())
  {
    std::cerr << "ROS master is not available, continue withtout ROS" << std::endl;
    return false;
  }
  return true;
}

geometry_msgs::TransformStamped PT2TF(const sva::PTransformd & X, const ros::Time & tm, const std::string & from, const std::string & to, unsigned int seq)
{
  geometry_msgs::TransformStamped msg;
  msg.header.seq = seq;
  msg.header.stamp = tm;
  msg.header.frame_id = from;
  msg.child_frame_id = to;

  Eigen::Vector4d q = Eigen::Quaterniond(X.rotation()).inverse().coeffs();
  const Eigen::Vector3d & t = X.translation();

  msg.transform.translation.x = t.x();
  msg.transform.translation.y = t.y();
  msg.transform.translation.z = t.z();

  msg.transform.rotation.w = q.w();
  msg.transform.rotation.x = q.x();
  msg.transform.rotation.y = q.y();
  msg.transform.rotation.z = q.z();

  return msg;
}

}

struct RobotPublisherImpl
{
  RobotPublisherImpl(const std::string & prefix)
  : nh(ros_init() ? new ros::NodeHandle() : 0),
    prefix_(prefix)
  {
    if(nh)
    {
      j_state_pub = nh->advertise<sensor_msgs::JointState>(prefix + "joint_states", 1);
      tf_caster.reset(new tf2_ros::TransformBroadcaster());
    }
  }

  void publish(const tvm::Robot & robot)
  {
    if(!nh) { return; }
    auto tm = ros::Time::now();
    sensor_msgs::JointState msg;
    msg.header.seq = seq_;
    msg.header.stamp = tm;
    msg.header.frame_id = "";
    msg.name.reserve(robot.mb().nrJoints() - 1);
    msg.position.reserve(robot.mb().nrJoints() - 1);
    msg.velocity.reserve(robot.mb().nrJoints() - 1);
    msg.effort.reserve(robot.mb().nrJoints() - 1);
    for(const auto & j : robot.mb().joints())
    {
      if(j.dof() == 1)
      {
        auto jIdx = robot.mb().jointIndexByName(j.name());
        if(robot.mbc().q[jIdx].size() > 0)
        {
          msg.name.push_back(j.name());
          msg.position.push_back(robot.mbc().q[jIdx][0]);
          msg.velocity.push_back(robot.mbc().alpha[jIdx][0]);
          msg.effort.push_back(robot.mbc().jointTorque[jIdx][0]);
        }
      }
    }

    std::vector<geometry_msgs::TransformStamped> tfs;
    tfs.push_back(PT2TF(robot.bodyTransform(robot.mb().body(0).name())*robot.mbc().parentToSon[0], tm, std::string("robot_map"), prefix_+robot.mb().body(0).name(), seq_));
    for(int j = 1; j < robot.mb().nrJoints(); ++j)
    {
      const auto & predIndex = robot.mb().predecessor(j);
      const auto & succIndex = robot.mb().successor(j);
      const auto & predName = robot.mb().body(predIndex).name();
      const auto & succName = robot.mb().body(succIndex).name();
      const auto & X_predp_pred = robot.bodyTransform(predName);
      const auto & X_succp_succ = robot.bodyTransform(succName);
      tfs.push_back(PT2TF(X_succp_succ*robot.mbc().parentToSon[static_cast<unsigned int>(j)]*X_predp_pred.inv(), tm, prefix_ + predName, prefix_ + succName, seq_));
    }
    seq_++;
    j_state_pub.publish(msg);
    tf_caster->sendTransform(tfs);
  }

  std::unique_ptr<ros::NodeHandle> nh;
  unsigned int seq_ = 0;
  std::string prefix_;
  ros::Publisher j_state_pub;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_caster;
};

RobotPublisher::RobotPublisher(const std::string & prefix)
: impl_(new RobotPublisherImpl(prefix))
{
}

RobotPublisher::~RobotPublisher()
{
}

void RobotPublisher::publish(const tvm::Robot & robot)
{
  impl_->publish(robot);
}
