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


void testDataGraph()
{
  auto robot = std::make_shared<RobotMockup>();
  auto f1 = std::make_shared<SomeRobotFunction1>(robot);
  auto user = std::make_shared<DataUser>();
  user->addInput(f1, { SomeRobotFunction1::Output::Value,
                      SomeRobotFunction1::Output::Jacobian,
                      SomeRobotFunction1::Output::Velocity,
                      SomeRobotFunction1::Output::NormalAcceleration,
                      SomeRobotFunction1::Output::JDot });

  UpdateGraph g;
  g.add(f1);
}

int main()
{
  //testVariable();
  testDataGraph();

  system("pause");
}