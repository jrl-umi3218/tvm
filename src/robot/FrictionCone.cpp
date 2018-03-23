#include <tvm/robot/internal/FrictionCone.h>

#include <Eigen/Geometry>

namespace tvm
{

namespace robot
{

namespace internal
{

FrictionCone::FrictionCone(const Eigen::Matrix3d & frame,
                           unsigned int nrGen, double mu, double dir)
: generators(nrGen)
{
  Eigen::Vector3d normal = frame.row(2);
  Eigen::Vector3d tan = dir*frame.row(0);
  double angle = std::atan(mu);

  Eigen::Vector3d gen = Eigen::AngleAxisd(angle, tan)*normal;
  double step = M_PI*2./nrGen;

  for(unsigned int i = 0; i < nrGen; ++i)
  {
    generators[i] = Eigen::AngleAxisd(dir*step*i, normal)*gen;
  }
}

}

}

}
