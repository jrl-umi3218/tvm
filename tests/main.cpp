#include "Variable.h"
#include "Mockup.h"

#include <iostream>

using namespace taskvm;

void testVariable()
{
  std::shared_ptr<Variable> v = Space(3).createVariable("v");
  std::cout << v->name() << std::endl;
  auto dv = dot(v);
  std::cout << dv->name() << std::endl;
  auto dv3 = dot(dv, 2);
  std::cout << dv3->name() << std::endl;
  auto dv5 = dot(v, 5);
  std::cout << dv5->name() << std::endl;
  auto dv4 = dot(dv, 3);
  std::cout << dv4->name() << std::endl;

  std::cout << " -------- " << std::endl;
  std::cout << (v == dv5->basePrimitive()) << std::endl;
  std::cout << (dv3 == dv4->primitive()) << std::endl;
}


void testDataGraphSimple()
{
  auto robot = std::make_shared<RobotMockup>();
  auto f1 = std::make_shared<SomeRobotFunction1>(robot);
  auto user = std::make_shared<DataUser>();
  user->addInput(f1, { SomeRobotFunction1::Output::Value,
                      SomeRobotFunction1::Output::Jacobian,
                      SomeRobotFunction1::Output::Velocity,
                      SomeRobotFunction1::Output::NormalAcceleration,
                      SomeRobotFunction1::Output::JDot });


  std::cout << "g0: adding only the robot" << std::endl;
  UpdateGraph g0;
  g0.add(robot);
  UpdatePlan p0(g0);
  p0.execute();
  std::cout << "p0 is empty. This is normal: as a user, robot does not depend on any update." << std::endl;

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g1: f1" << std::endl;
  UpdateGraph g1;
  g1.add(f1);
  UpdatePlan p1(g1);
  p1.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g2: user" << std::endl;
  UpdateGraph g2;
  g2.add(user);
  UpdatePlan p2(g2);
  p2.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g3: f1, user" << std::endl;
  UpdateGraph g3;
  g3.add(f1);
  g3.add(user);
  UpdatePlan p3(g3);
  p3.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g4: user, f1" << std::endl;
  UpdateGraph g4;
  g4.add(user);
  g4.add(f1);
  UpdatePlan p4(g4);
  p4.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;
}

void testDataGraphComplex()
{
  auto robot = std::make_shared<RobotMockup>();
  auto f1 = std::make_shared<SomeRobotFunction1>(robot);
  auto f2 = std::make_shared<SomeRobotFunction2>(robot);

  //IK-like
  auto cik1 = std::make_shared<KinematicLinearizedConstraint>("Linearized f1", f1);
  auto cik2 = std::make_shared<KinematicLinearizedConstraint>("Linearized f2", f2);

  auto userIK = std::make_shared<DataUser>();
  userIK->addInput(cik1, { LinearConstraint::Output::A, LinearConstraint::Output::b });
  userIK->addInput(cik2, { LinearConstraint::Output::A, LinearConstraint::Output::b });

  UpdateGraph gik;
  gik.add(userIK);
  UpdatePlan pik(gik);
  pik.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  //ID-like
  auto cid0 = std::make_shared<DynamicEquation>("EoM", robot);
  auto cid1 = std::make_shared<DynamicLinearizedConstraint>("Linearized f1", f1);
  auto cid2 = std::make_shared<DynamicLinearizedConstraint>("Linearized f2", f2);

  auto userID = std::make_shared<DataUser>();
  userID->addInput(cid0, { LinearConstraint::Output::A, LinearConstraint::Output::b });
  userID->addInput(cid1, { LinearConstraint::Output::A, LinearConstraint::Output::b });
  userID->addInput(cid2, { LinearConstraint::Output::A, LinearConstraint::Output::b });

  UpdateGraph gid;
  gid.add(userID);
  UpdatePlan pid(gid);
  pid.execute();
}

int main()
{
  //testVariable();
  testDataGraphSimple();
  testDataGraphComplex();

#ifdef WIN32
  system("pause");
#endif
}
