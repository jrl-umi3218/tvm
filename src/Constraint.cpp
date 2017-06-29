#include "Constraint.h"

const Eigen::VectorXd & taskvm::Constraint::l() const
{
  return l_;
}

Eigen::VectorXd & taskvm::Constraint::l()
{
  return l_;
}

const Eigen::VectorXd & taskvm::Constraint::u() const
{
  return u_;
}

Eigen::VectorXd & taskvm::Constraint::u()
{
  return u_;
}
