#include <tvm/data/Node.h>
#include <tvm/data/OutputSelector.h>

#include <iostream>

using namespace tvm::data;

class Provider1 : public Node<Provider1>
{
public:
  SET_OUTPUTS(Provider1, O0, O1, O2)

  Provider1()
  {
    std::cout << " - adding O0 O1 O2" << std::endl;
  }
};

class Provider2 : public Provider1
{
public:
  SET_OUTPUTS(Provider2, O3, O4)
  DISABLE_OUTPUTS(Provider1::Output::O1)

  Provider2()
  {
    std::cout << " - adding O3 O4" << std::endl;
    std::cout << " - statically disabling O1" << std::endl;
  }
};

class Provider3 : public OutputSelector<Provider2>
{
public:
  Provider3()
  {
    disableOutput(Provider1::Output::O1, Provider2::Output::O3);
    std::cout << " - dynamically disabling O1 and O3" << std::endl;
  }
};

class Provider4 : public Provider3
{
public:
  SET_OUTPUTS(Provider4, O5, O6, O7)

  Provider4()
  {
    std::cout << " - adding O5 O6 O7" << std::endl;
  }
};

class Provider5 : public OutputSelector<Provider4>
{
public:
  Provider5()
  {
    disableOutput(Provider1::Output::O0, Provider4::Output::O6);
    std::cout << " - dynamically disabling O0 and O6" << std::endl;
  }

  template<typename EnumT>
  void manualEnable(EnumT e) { enableOutput(e); }

  template<typename EnumT>
  void manualDisable(EnumT e) { disableOutput(e); }
};


void outputSelectorTest()
{
  std::cout << "Building Provider3" << std::endl;
  Provider3 p3;
  std::cout << std::endl;
  std::cout << "O0: " << p3.isOutputEnabled(Provider1::Output::O0) << std::endl;
  std::cout << "O1: " << p3.isOutputEnabled(Provider1::Output::O1) << std::endl;
  std::cout << "O2: " << p3.isOutputEnabled(Provider1::Output::O2) << std::endl;
  std::cout << "O3: " << p3.isOutputEnabled(Provider2::Output::O3) << std::endl;
  std::cout << "O4: " << p3.isOutputEnabled(Provider2::Output::O4) << std::endl;
  std::cout << std::endl;

  std::cout << "Building Provider5" << std::endl;
  Provider5 p5;
  std::cout << std::endl;
  for (int i = 0; i < 8; ++i)
    std::cout << "O" << i << ": " << p5.isOutputEnabled(i) << std::endl;

  p5.manualEnable(Provider4::Output::O6);
  try { p5.manualEnable(Provider1::Output::O1); }
  catch (std::exception e) { std::cout << "catched exception: " << e.what() << std::endl; }
  p5.lock();
  try { p5.manualDisable(Provider1::Output::O2); }
  catch (std::exception e) { std::cout << "catched exception: " << e.what() << std::endl; }
}