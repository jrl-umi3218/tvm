#include "Variable.h"
#include "Mockup.h"

#include "tvm/data/Node.h"
#include "tvm/CallGraph.h"

#include <iostream>

void testCompiledAssignment();
void testCompiledAssignmentWrapper();
void outputSelectorTest();
void linearConstraintTest();
void testSolvingRequirements();

using namespace tvm;

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
  auto user = std::make_shared<tvm::data::Inputs>();
  user->addInput(f1, SomeRobotFunction1::Output::Value,
                     SomeRobotFunction1::Output::Jacobian,
                     SomeRobotFunction1::Output::Velocity,
                     SomeRobotFunction1::Output::NormalAcceleration,
                     SomeRobotFunction1::Output::JDot);


  std::cout << "g0: adding only the robot" << std::endl;
  tvm::CallGraph g0;
  g0.add(robot);
  g0.update();
  g0.execute();
  std::cout << "g0 execution is empty. This is normal: as a user, robot does not depend on any update." << std::endl;

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g1: f1" << std::endl;
  tvm::CallGraph g1;
  g1.add(f1);
  g1.update();
  g1.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g2: user" << std::endl;
  tvm::CallGraph g2;
  g2.add(user);
  g2.update();
  g2.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g3: f1, user" << std::endl;
  tvm::CallGraph g3;
  g3.add(f1);
  g3.add(user);
  g3.update();
  g3.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  std::cout << "g4: user, f1" << std::endl;
  tvm::CallGraph g4;
  g4.add(user);
  g4.add(f1);
  g4.update();
  g4.execute();

  std::cout << std::endl << "----------------------" << std::endl << std::endl;


}

void testBadGraph()
{
  auto robot = std::make_shared<RobotMockup>();
  auto f1 = std::make_shared<SomeRobotFunction2>(robot);
  auto f2 = std::make_shared<BadRobotFunction>(robot);
  auto user1 = std::make_shared<tvm::data::Inputs>();
  try
  {
    user1->addInput(f1, SomeRobotFunction2::Output::Value,
                        SomeRobotFunction2::Output::Jacobian,
                        SomeRobotFunction2::Output::Velocity,
                        SomeRobotFunction2::Output::NormalAcceleration,
                        SomeRobotFunction2::Output::JDot);
  }
  catch (std::exception e)
  {
    std::cout << e.what() << std::endl;
  }

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  auto user2 = std::make_shared<tvm::data::Inputs>();
  user2->addInput(f2, BadRobotFunction::Output::Jacobian);

  tvm::CallGraph g2;
  g2.add(user2);
  try { g2.update(); }
  catch (std::exception & e) { std::cout << "Got exception: " << e.what() << std::endl; }

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

  auto cik = std::make_shared<KinematicLinearizedConstraint>("IK", f2);
  auto user3 = std::make_shared<tvm::data::Inputs>();
  user3->addInput(cik, LinearConstraint::Output::A, LinearConstraint::Output::b);
  tvm::CallGraph g3;
  g3.add(user3);
  try { g3.update(); }
  catch (std::exception & e) { std::cout << "Got exception: " << e.what() << std::endl; }

  std::cout << std::endl << "----------------------" << std::endl << std::endl;

}

void testDataGraphComplex()
{
  {
    auto robot = std::make_shared<RobotMockup>();
    auto f1 = std::make_shared<SomeRobotFunction1>(robot);
    auto f2 = std::make_shared<SomeRobotFunction2>(robot);

    //IK-like
    auto cik1 = std::make_shared<KinematicLinearizedConstraint>("Linearized f1", f1);
    auto cik2 = std::make_shared<KinematicLinearizedConstraint>("Linearized f2", f2);

    auto userIK = std::make_shared<tvm::data::Inputs>();
    userIK->addInput(cik1, LinearConstraint::Output::A, LinearConstraint::Output::b);
    userIK->addInput(cik2, LinearConstraint::Output::A, LinearConstraint::Output::b);

    tvm::CallGraph gik;
    gik.add(userIK);
    gik.update();
    gik.execute();

    std::cout << std::endl << "----------------------" << std::endl << std::endl;

    //ID-like
    auto cid0 = std::make_shared<DynamicEquation>("EoM", robot);
    auto cid1 = std::make_shared<DynamicLinearizedConstraint>("Linearized f1", f1);

    auto userID = std::make_shared<tvm::data::Inputs>();
    userID->addInput(cid0, LinearConstraint::Output::A, LinearConstraint::Output::b);
    userID->addInput(cid1, LinearConstraint::Output::A, LinearConstraint::Output::b);

    tvm::CallGraph gid;
    gid.add(userID);
    gid.update();
    gid.execute();
  }
  std::cout << std::endl << std::endl << "Now with direct dependency" << std::endl << std::endl;
  {
    auto robot = std::make_shared<RobotMockup>();
    auto f1 = std::make_shared<SomeRobotFunction1>(robot);
    auto f2 = std::make_shared<SomeRobotFunction2>(robot);

    //IK-like
    auto cik1 = std::make_shared<BetterKinematicLinearizedConstraint>("Linearized f1", f1);
    auto cik2 = std::make_shared<BetterKinematicLinearizedConstraint>("Linearized f2", f2);

    auto userIK = std::make_shared<tvm::data::Inputs>();
    userIK->addInput(cik1, BetterLinearConstraint::Output::A, BetterLinearConstraint::Output::b);
    userIK->addInput(cik2, BetterLinearConstraint::Output::A, BetterLinearConstraint::Output::b);

    tvm::CallGraph gik;
    gik.add(userIK);
    gik.update();
    gik.execute();

    std::cout << std::endl << "----------------------" << std::endl << std::endl;

    //ID-like
    auto cid0 = std::make_shared<DynamicEquation>("EoM", robot);
    auto cid1 = std::make_shared<BetterDynamicLinearizedConstraint>("Linearized f1", f1);

    auto userID = std::make_shared<tvm::data::Inputs>();
    userID->addInput(cid0, LinearConstraint::Output::A, LinearConstraint::Output::b);
    userID->addInput(cid1, BetterLinearConstraint::Output::A, BetterLinearConstraint::Output::b);

    tvm::CallGraph gid;
    gid.add(userID);
    gid.update();
    gid.execute();
  }
}

int main()
{
  //testVariable();
  //testDataGraphSimple();
  //testBadGraph();
  //testDataGraphComplex();
  //outputSelectorTest();
  //linearConstraintTest();
  //testSolvingRequirements();


  //testCompiledAssignment();
  testCompiledAssignmentWrapper();

#ifdef WIN32
  system("pause");
#endif
}
