#include <tvm/data/Node.h>
#include <tvm/data/OutputSelector.h>

// boost
#define BOOST_TEST_MODULE OutputSelectorTest
#include <boost/test/unit_test.hpp>

#include <iostream>

using namespace tvm::data;

class Provider1 : public Node<Provider1>
{
public:
  SET_OUTPUTS(Provider1, O0, O1, O2)

  Provider1()
  {
  }
};

class Provider2 : public Provider1
{
public:
  SET_OUTPUTS(Provider2, O3, O4)
  DISABLE_OUTPUTS(Provider1::Output::O1)

  Provider2()
  {
  }
};

class Provider3 : public OutputSelector<Provider2>
{
public:
  Provider3()
  {
    disableOutput(Provider1::Output::O1, Provider2::Output::O3);
  }
};

class Provider4 : public Provider3
{
public:
  SET_OUTPUTS(Provider4, O5, O6, O7)

  Provider4()
  {
  }
};

class Provider5 : public OutputSelector<Provider4>
{
public:
  Provider5()
  {
    disableOutput(Provider1::Output::O0, Provider4::Output::O6);
  }

  template<typename EnumT>
  void manualEnable(EnumT e) { enableOutput(e); }

  template<typename EnumT>
  void manualDisable(EnumT e) { disableOutput(e); }
};


BOOST_AUTO_TEST_CASE(OutputSelectorTest)
{
  Provider3 p3;
  BOOST_CHECK(p3.isOutputEnabled(Provider1::Output::O0));
  BOOST_CHECK(!p3.isOutputEnabled(Provider1::Output::O1));
  BOOST_CHECK(p3.isOutputEnabled(Provider1::Output::O2));
  BOOST_CHECK(!p3.isOutputEnabled(Provider2::Output::O3));
  BOOST_CHECK(p3.isOutputEnabled(Provider2::Output::O4));

  Provider5 p5;
  for (int i = 0; i < 8; ++i)
  {
    if(i != 0 && i != 1 && i != 3 && i != 6)
    {
      BOOST_CHECK(p5.isOutputEnabled(i));
    }
    else
    {
      BOOST_CHECK(!p5.isOutputEnabled(i));
    }
  }

  p5.manualEnable(Provider4::Output::O6);
  BOOST_CHECK_THROW(p5.manualEnable(Provider1::Output::O1), std::runtime_error);
  p5.lock();
  BOOST_CHECK_THROW(p5.manualDisable(Provider1::Output::O2), std::runtime_error);
}
