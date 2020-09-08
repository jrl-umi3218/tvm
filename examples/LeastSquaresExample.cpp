// Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM

#include <tvm/LinearizedControlProblem.h>
#include <tvm/Variable.h>
#include <tvm/scheme/WeightedLeastSquares.h>
#include <tvm/solver/defaultLeastSquareSolver.h>

using namespace tvm;
using namespace requirements;
using namespace Eigen;

// Constructs a rotation matrix bringing the unit vector z = (0,0,1)^T to v
// Adapted from https://math.stackexchange.com/a/476311
Matrix3d rotationFromZ(Vector3d v)
{
  v.normalize();
  Matrix3d R;
  if(v[2] + 1 < 1e-8)
  {
    // clang-format off
    R << 1, 0, 0,
         0, 1, 0,
         0, 0, -1;
    // clang-format on
  }
  else
  {
    double a = 1 / (1 + v[2]);
    double ax2 = a * v[0] * v[0];
    double ay2 = a * v[1] * v[1];
    double axy = a * v[0] * v[1];
    // clang-format off
    R << 1 - ax2,   -axy , v[0],
         -axy   , 1 - ay2, v[1],
          -v[0] ,  -v[1] , v[2];
    // clang-format on
  }
  return R;
}

// Returns the antisymmetric matrix S such that for a fector x, S*x = v.cross(x)
Matrix3d hat(const Vector3d & v)
{
  Matrix3d S;
  // clang-format off
  S <<  0  , -v[2],  v[1],
       v[2],   0  , -v[0],
      -v[1],  v[0],   0  ;
  // clang-format on
  return S;
}

// Returns a four-sided linearized cone with axis z = (0,0,1)^T and friction coefficient mu
MatrixXd discretizedFrictionCone(double mu)
{
  double mub = mu / std::sqrt(2);
  MatrixXd C(4, 3);
  // clang-format off
  C << Matrix2d::Identity(), Vector2d::Constant(mub),
      -Matrix2d::Identity(), Vector2d::Constant(mub);
  // clang-format on
  return C;
}

bool leastSquares3points()
{
  // Creating a space R^3
  Space S(3);

  // Data for first contact point
  Vector3d p1(-1, -1, 0);                  // Contact point position
  Vector3d n1(1, 1, 1);                    // Normal vector to contact surface
  MatrixXd R1 = rotationFromZ(n1);         // Rotation between world frame and a local contact frame
  Vector3d f1des = Vector3d::Zero();       // Desired force for f1
  VariablePtr f1 = S.createVariable("f1"); // Variable creation

  // Data for second contact point
  Vector3d p2(1, 0, 0.5);
  Vector3d n2(-1, 0, 1);
  MatrixXd R2 = rotationFromZ(n2);
  Vector3d f2des = Vector3d::Zero();
  VariablePtr f2 = S.createVariable("f2");

  // Data for third contact point
  Vector3d p3(0, 1, 1);
  Vector3d n3(0, -1, 0);
  MatrixXd R3 = rotationFromZ(n3);
  Vector3d f3des = Vector3d::Zero();
  VariablePtr f3 = S.createVariable("f3");

  const double m = 1;                    // Mass
  const Vector3d g(0, 0, -9.81);         // Gravity vector
  VariablePtr c = S.createVariable("c"); // Center of mass variable

  MatrixXd C = discretizedFrictionCone(0.6); // Matrix of a discretized cone with friction coefficient 0.6

  // Creating the problem
  LinearizedControlProblem pb;
  pb.add(R1 * f1 + R2 * f2 + R3 * f3 + m * g == 0., PriorityLevel(0)); // Newton equation
  pb.add(hat(p1) * R1 * f1 + hat(p2) * R2 * f2 + hat(p3) * R3 * f3 + m * hat(g) * c == 0.,
         PriorityLevel(0));               // Euler equation
  pb.add(C * f1 >= 0., PriorityLevel(0)); // Friction cone constraints
  pb.add(C * f2 >= 0., PriorityLevel(0));
  pb.add(C * f3 >= 0., PriorityLevel(0));
  pb.add(f1 == f1des, PriorityLevel(1)); // Desired forces
  pb.add(f2 == f2des, PriorityLevel(1));
  pb.add(f3 == f3des, PriorityLevel(1));
  pb.add(c == p1, {PriorityLevel(1), Weight(1e-4)}); // Desired CoM position
  pb.add(c == p2, {PriorityLevel(1), Weight(1e-4)});
  pb.add(c == p3, {PriorityLevel(1), Weight(1e-4)});

  // Creating the resolution scheme
  scheme::WeightedLeastSquares solver(solver::DefaultLSSolverOptions().verbose(true));

  // And solving the problem
  bool found = solver.solve(pb);
  return found;
}

// Let's run some quick tests to ensure this example is not outdated and compiles.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

TEST_CASE("Running least-squares example")
{
  bool solved = leastSquares3points();
  FAST_CHECK_UNARY(solved);
}
