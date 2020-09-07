/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/graph/abstract/Node.h>
#include <tvm/graph/abstract/OutputSelector.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#include <iostream>

using namespace tvm::graph::abstract;

class Provider1 : public Node<Provider1>
{
public:
  SET_OUTPUTS(Provider1, O0, O1, O2)

  Provider1() {}
};

class Provider2 : public Provider1
{
public:
  SET_OUTPUTS(Provider2, O3, O4)
  DISABLE_OUTPUTS(Provider1::Output::O1)

  Provider2() {}
};

class Provider3 : public OutputSelector<Provider2>
{
public:
  Provider3() { disableOutput(Provider1::Output::O1, Provider2::Output::O3); }
};

class Provider4 : public Provider3
{
public:
  SET_OUTPUTS(Provider4, O5, O6, O7)

  Provider4() {}
};

class Provider5 : public OutputSelector<Provider4>
{
public:
  Provider5() { disableOutput(Provider1::Output::O0, Provider4::Output::O6); }

  template<typename EnumT>
  void manualEnable(EnumT e)
  {
    enableOutput(e);
  }

  template<typename EnumT>
  void manualDisable(EnumT e)
  {
    disableOutput(e);
  }
};

class Provider6 : public OutputSelector<Provider6, Provider4>
{
public:
  SET_OUTPUTS(Provider6, O8, O9)

  Provider6()
  {
    disableOutput(Provider6::Output::O9, Provider4::Output::O5, Provider2::Output::O3, Provider1::Output::O1);
  }
};

TEST_CASE("Test outputs selector")
{
  Provider3 p3;
  FAST_CHECK_UNARY(p3.isOutputEnabled(Provider1::Output::O0));
  FAST_CHECK_UNARY_FALSE(p3.isOutputEnabled(Provider1::Output::O1));
  FAST_CHECK_UNARY(p3.isOutputEnabled(Provider1::Output::O2));
  FAST_CHECK_UNARY_FALSE(p3.isOutputEnabled(Provider2::Output::O3));
  FAST_CHECK_UNARY(p3.isOutputEnabled(Provider2::Output::O4));

  Provider5 p5;
  for(int i = 0; i < 8; ++i)
  {
    if(i != 0 && i != 1 && i != 3 && i != 6)
    {
      FAST_CHECK_UNARY(p5.isOutputEnabled(i));
    }
    else
    {
      FAST_CHECK_UNARY_FALSE(p5.isOutputEnabled(i));
    }
  }

  Provider6 p6;
  for(int i = 0; i < 10; ++i)
  {
    if(i != 1 && i != 3 && i != 5 && i != 9)
    {
      FAST_CHECK_UNARY(p6.isOutputEnabled(i));
    }
    else
    {
      FAST_CHECK_UNARY_FALSE(p6.isOutputEnabled(i));
    }
  }

  p5.manualEnable(Provider4::Output::O6);
  CHECK_THROWS_AS(p5.manualEnable(Provider1::Output::O1), std::runtime_error);
  p5.lock();
  CHECK_THROWS_AS(p5.manualDisable(Provider1::Output::O2), std::runtime_error);
}
