#include <tvm/data/Node.h>

#include <iostream>

struct FailedTest {};

struct RegisterDisabledUpdate : public tvm::data::Node<RegisterDisabledUpdate>
{
  SET_UPDATES(RegisterDisabledUpdate, U0, U1)
  DISABLE_UPDATES(Update::U0)

  RegisterDisabledUpdate()
  {
    registerUpdates(Update::U0, &RegisterDisabledUpdate::updateU0);
  }

  void updateU0() {}
};

struct DisabledOutputDependency : public tvm::data::Node<DisabledOutputDependency>
{
  SET_OUTPUTS(DisabledOutputDependency, O0, O1)
  DISABLE_OUTPUTS(Output::O0)
  SET_UPDATES(DisabledOutputDependency, U0)

  DisabledOutputDependency()
  {
    addOutputDependency({Output::O0, Output::O1}, Update::U0);
  }
};

struct DisabledInternalDependency : public tvm::data::Node<DisabledInternalDependency>
{
  SET_UPDATES(DisabledInternalDependency, U0, U1)
  DISABLE_UPDATES(Update::U0)

  DisabledInternalDependency()
  {
    addInternalDependency(Update::U0, Update::U1);
  }
};

struct DisabledOutput : public tvm::data::Node<DisabledOutput>
{
  SET_OUTPUTS(DisabledOutput, O0, O1)
  DISABLE_OUTPUTS(Output::O0)
};

struct DisabledUpdateInputDependency : public tvm::data::Node<DisabledUpdateInputDependency>
{
  SET_UPDATES(DisabledUpdateInputDependency, U0)
  DISABLE_UPDATES(Update::U0)
  DisabledUpdateInputDependency(std::shared_ptr<DisabledOutput> s)
  {
    addInputDependency(Update::U0, s, DisabledOutput::Output::O1);
  }
};

struct DisabledOutputInputDependency : public tvm::data::Node<DisabledOutputInputDependency>
{
  SET_UPDATES(DisabledOutputInputDependency, U0)
  DisabledOutputInputDependency(std::shared_ptr<DisabledOutput> s)
  {
    addInputDependency(Update::U0, s, DisabledOutput::Output::O0);
  }
};

#define TEST(T, ...) \
  try { T obj{__VA_ARGS__}; throw FailedTest(); } catch(std::runtime_error & exc) { std::cout << #T" threw runtime_error: " << exc.what() << std::endl; }

int main()
{
  TEST(RegisterDisabledUpdate)
  TEST(DisabledOutputDependency)
  TEST(DisabledInternalDependency)
  TEST(DisabledUpdateInputDependency, std::make_shared<DisabledOutput>())
  TEST(DisabledOutputInputDependency, std::make_shared<DisabledOutput>())
  return 0;
}
