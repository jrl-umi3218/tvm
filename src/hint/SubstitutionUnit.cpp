/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/Variable.h>
#include <tvm/hint/internal/GenericCalculator.h>
#include <tvm/hint/internal/SubstitutionUnit.h>
#include <tvm/utils/memoryChecks.h>

#include <algorithm>
#include <set>

namespace
{
using namespace tvm;
using namespace tvm::hint;

/** Return a single substitution grouping the substitutions from \p substitutionPool
 * specified by \p group.
 */
Substitution group(const std::vector<Substitution> & substitutionPool, const std::vector<size_t> & group)
{
  std::set<LinearConstraintPtr> subs;
  std::set<VariablePtr> vars;

  for(auto i : group)
  {
    const auto & c = substitutionPool[i].constraints();
    subs.insert(c.begin(), c.end());

    const auto & v = substitutionPool[i].variables();
    vars.insert(v.begin(), v.end());
  }

  // determine the rank.
  // We make two suppositions:
  // - the substitutions are independent
  // - if x1,...,xk are the variables susbtituted by a substitution, and
  //   xk+1,... are the variables substituted by other substitutions, the
  //   matrices in front of xk+1,... are independent of the matrix in front
  //   of x1,...,xk.
  //  If this is not the case, an exception will be raised later in the
  //  calculator.
  int rank = 0;
  for(auto i : group)
  {
    const auto & s = substitutionPool[i];
    const auto & v = s.variables();
    int ranki = s.rank(); // this is the declared rank of the matrix in front
                          // of x1,...,xk
    for(const auto & xi : vars)
    {
      int mi = 0;
      // xi is one of the substituted variables. If it is not one
      // substituted by s...
      if(std::find(v.begin(), v.end(), xi) == v.end())
      {
        for(const auto & c : s.constraints())
        {
          if(c->variables().contains(*xi))
            mi += c->size();
        }
        // it can help to increase the rank
        ranki += std::min(mi, xi->size());
      }
    }
    rank += std::min(ranki, s.m());
  }

  return Substitution(std::vector<LinearConstraintPtr>(subs.begin(), subs.end()),
                      std::vector<VariablePtr>(vars.begin(), vars.end()), rank);
}
} // namespace

namespace tvm
{
namespace hint
{
namespace internal
{
SubstitutionUnit::SubstitutionUnit(const std::vector<Substitution> & substitutionPool,
                                   const std::vector<std::vector<size_t>> & groups,
                                   const std::vector<size_t> order)
{
  extractSubstitutions(substitutionPool, groups, order);
  scanSubstitutions();
  computeDependencies();
  tvm::utils::override_is_malloc_allowed(true);
  initializeMatrices();
  createFunctions();
  tvm::utils::restore_is_malloc_allowed();
}

void SubstitutionUnit::update()
{
  for(const auto & c : calculators_)
  {
    c->update();
  }
  int m = 0;
  // Knowing that the substitutions are ordered we process as follows, we process
  // one substitution at a time, in order.
  // For each constraint sum A_{ij} x_j + sum B_{ij} y_j = c of substitution k,
  // we first fill the matrices B_{ij}.
  // We then substitutes the variable x_l that are not the ones substituted by
  // this substitution by x_l = sum M_{lj} y_j + sum AsZ_{lj} z_j + u_l. Since the
  // substitutions are ordered, we know that the values M_{lj}, AsZ_{lj} and u_l
  // have been computed at a previous iteration k.
  // Once this has been done for every constraints in the substitution, we compute
  // M, AsZ, u, S^T B, S^T Z and S^T c corresponding to the concatenation of the
  // variables substituted by the substitution.
  for(size_t k = 0; k < substitutions_.size(); ++k)
  {
    int mk = 0;
    bool cZero = true; // true if cIsZero_[i] for all i in sub2cstr_[k]
    for(auto i : sub2cstr_[k])
    {
      std::fill(firstY_.begin(), firstY_.end(), true);
      std::fill(firstZ_.begin(), firstZ_.end(), true);
      const auto & c = constraints_[i];
      auto mki = c->size();
      for(auto j : constraintsY_[i])
      {
        auto r = y_[j]->getMappingIn(y_);
        B_.block(m + mk, r.start, mki, r.dim) = c->jacobian(*y_[j]);
        firstY_[j] = false;
      }
      switch(c->rhs())
      {
        case constraint::RHS::ZERO:
          c_.segment(m + mk, mki).setZero();
          cIsZero_[i] = true;
          break;
        case constraint::RHS::AS_GIVEN:
          c_.segment(m + mk, mki) = c->e();
          cIsZero_[i] = false;
          break;
        case constraint::RHS::OPPOSITE:
          c_.segment(m + mk, mki) = -c->e();
          cIsZero_[i] = false;
          break;
      }
      for(auto l : CXdependencies_[i])
      {
        auto rx = x_[l]->getMappingIn(x_);
        const auto & A = c->jacobian(*x_[l]);
        // B_{ij} += A_{il}*M_{lj} for each y_j on which x_l depends.
        for(auto j : XYdependencies_[l])
        {
          auto ry = y_[j]->getMappingIn(y_);
          if(firstY_[j])
          {
            B_.block(m + mk, ry.start, mki, ry.dim).noalias() = A * M_.block(rx.start, ry.start, rx.dim, ry.dim);
            firstY_[j] = false;
          }
          else
          {
            B_.block(m + mk, ry.start, mki, ry.dim).noalias() += A * M_.block(rx.start, ry.start, rx.dim, ry.dim);
          }
        }
        // Z_{ij} += A_{il}*AsZ_{lj} for each z_j on which x_l depends.
        for(auto j : XZdependencies_[l])
        {
          auto rz = z_[j]->getMappingIn(z_);
          if(j == static_cast<int>(x2sub_[l]))
          {
            // we handles this dependency in z separately, as it involves N
            calculators_[j]->postMultiplyByN(Z_.block(m + mk, rz.start, mki, rz.dim), A, xRange_[l], !firstZ_[j]);
            firstZ_[j] = false;
          }
          else if(firstZ_[j])
          {
            Z_.block(m + mk, rz.start, mki, rz.dim).noalias() = A * AsZ_.block(rx.start, rz.start, rx.dim, rz.dim);
            firstZ_[j] = false;
          }
          else
          {
            Z_.block(m + mk, rz.start, mki, rz.dim).noalias() += A * AsZ_.block(rx.start, rz.start, rx.dim, rz.dim);
          }
        }

        // c_i -= A{il}*u_l
        c_.segment(m + mk, mki).noalias() -= A * u_.segment(rx.start, rx.dim);
        cIsZero_[i] = cIsZero_[i] && uIsZero_[l];
      }
      cZero = cZero && cIsZero_[i];
      mk += mki;
    }
    auto rn = substitutionNRanges_[k];
    // Compute M_{kj} and S_k^T B_{k,j} for y_[j] on which substitutions_[k] depends.
    for(auto j : SYdependencies_[k])
    {
      auto ry = y_[j]->getMappingIn(y_);
      calculators_[k]->premultiplyByASharpAndSTranspose(M_.block(rn.start, ry.start, rn.dim, ry.dim),
                                                        StB_[k].middleCols(ry.start, ry.dim),
                                                        B_.block(m, ry.start, mk, ry.dim), true);
    }
    // Compute AsZ_{kj} and S_k^T Z_{k,j} for z_[j] on which substitutions_[k] depends.
    for(auto j : SZdependencies_[k])
    {
      auto rz = z_[j]->getMappingIn(z_);
      calculators_[k]->premultiplyByASharpAndSTranspose(AsZ_.block(rn.start, rz.start, rn.dim, rz.dim),
                                                        StZ_[k].middleCols(rz.start, rz.dim),
                                                        Z_.block(m, rz.start, mk, rz.dim), true);
    }
    // Compute u_k and S_k^T c_k
    calculators_[k]->premultiplyByASharpAndSTranspose(u_.segment(rn.start, rn.dim), Stc_[k], c_.segment(m, mk), false);
    uIsZero_[k] = cZero;
    m += mk;
  }

  // update values in varSubstitution and remaining_
  for(size_t k = 0; k < substitutions_.size(); ++k)
  {
    for(auto i : sub2x_[k])
    {
      auto rx = x_[static_cast<int>(i)]->getMappingIn(x_);
      for(auto j : SYdependencies_[k])
      {
        auto ry = y_[j]->getMappingIn(y_);
        varSubstitutions_[i]->A(M_.block(rx.start, ry.start, rx.dim, ry.dim), *y_[j]);
      }
      for(auto j : SZdependencies_[k])
      {
        if(j == static_cast<int>(k))
          continue;
        auto rz = z_[j]->getMappingIn(z_);
        varSubstitutions_[i]->A(AsZ_.block(rx.start, rz.start, rx.dim, rz.dim), *z_[j]);
      }
      // copy N
      if(z_[static_cast<int>(k)]->size() > 0)
      {
        varSubstitutions_[i]->A(calculators_[k]->N().middleRows(xRange_[i].start, xRange_[i].dim),
                                *z_[static_cast<int>(k)]);
      }
      using tvm::internal::MatrixProperties;
      MatrixProperties p;
      if(uIsZero_[k])
      {
        p = {MatrixProperties::Constness(true), MatrixProperties::ZERO};
      }
      varSubstitutions_[i]->b(u_.segment(rx.start, rx.dim), p);
    }

    for(auto j : SYdependencies_[k])
    {
      auto ry = y_[j]->getMappingIn(y_);
      remaining_[k]->A(StB_[k].middleCols(ry.start, ry.dim), *y_[j]);
    }
    for(auto j : SZdependencies_[k])
    {
      auto rz = z_[j]->getMappingIn(z_);
      remaining_[k]->A(StZ_[k].middleCols(rz.start, rz.dim), *z_[j]);
    }
    remaining_[k]->b(Stc_[k]);
  }
}

const std::vector<VariablePtr> & SubstitutionUnit::variables() const { return x_.variables(); }

const std::vector<std::shared_ptr<function::BasicLinearFunction>> & SubstitutionUnit::variableSubstitutions() const
{
  return varSubstitutions_;
}

const std::vector<VariablePtr> & SubstitutionUnit::additionalVariables() const { return z_.variables(); }

const std::vector<std::shared_ptr<constraint::BasicLinearConstraint>> & SubstitutionUnit::additionalConstraints() const
{
  return remaining_;
}

const std::vector<VariablePtr> & SubstitutionUnit::otherVariables() const { return y_.variables(); }

void SubstitutionUnit::extractSubstitutions(const std::vector<Substitution> & substitutionPool,
                                            const std::vector<std::vector<size_t>> & groups,
                                            const std::vector<size_t> order)
{
  for(auto i : order)
  {
    assert(groups[i].size() > 0);
    if(groups[i].size() == 1)
    {
      substitutions_.push_back(substitutionPool[groups[i].front()]);
    }
    else
    {
      substitutions_.push_back(group(substitutionPool, groups[i]));
    }
  }
}

void SubstitutionUnit::scanSubstitutions()
{
  m_ = 0;
  int n = 0;
  for(size_t i = 0; i < substitutions_.size(); ++i)
  {
    const auto & s = substitutions_[i];
    calculators_.push_back(s.calculator());
    sub2cstr_.push_back({});
    sub2x_.push_back({});
    SYdependencies_.push_back({});
    SZdependencies_.push_back({});
    uIsZero_.push_back(false);
    int ni = 0;
    for(const auto & v : s.variables())
    {
      sub2x_[i].push_back(x_.variables().size());
      x2sub_.push_back(i);
      x_.add(v);
      xRange_.push_back({ni, v->size()});
      ni += v->size();
      XYdependencies_.push_back({});
      XZdependencies_.push_back({});
      cIsZero_.push_back(false);
    }
    int mi = 0;
    for(const auto c : s.constraints())
    {
      sub2cstr_[i].push_back(constraints_.size());
      constraints_.push_back(c);
      constraintsY_.push_back({});
      CXdependencies_.push_back({});
      mi += c->size();
      for(const auto & v : c->variables())
      {
        // Is v a variable to be substituted? It is ok to test only with an
        // incomplete vars_, as the susbtitution are supposed to be given in
        // a correct order.
        auto it = std::find(x_.variables().begin(), x_.variables().end(), v);
        if(it == x_.variables().end())
        {
          // v is an y variable.
          y_.add(v);
          auto ity = std::find(y_.variables().begin(), y_.variables().end(), v);
          constraintsY_.back().push_back(static_cast<int>(ity - y_.variables().begin()));
        }
        else if(std::find(s.variables().begin(), s.variables().end(), v) == s.variables().end())
        {
          // v is an x variable that is not substituted by s
          CXdependencies_.back().push_back(static_cast<int>(it - x_.variables().begin()));
        }
      }
    }
    std::stringstream ss;
    ss << "z" << i;
    z_.add(Space(ni - s.rank()).createVariable(ss.str()));
    substitutionMRanges_.push_back({m_, mi});
    substitutionNRanges_.push_back({n, ni});
    m_ += mi;
    n += ni;
  }
}

void SubstitutionUnit::initializeMatrices()
{
  B_.resize(m_, y_.totalSize());
  Z_.resize(m_, z_.totalSize());
  c_.resize(m_);
  M_.resize(x_.totalSize(), y_.totalSize());
  AsZ_.resize(x_.totalSize(), z_.totalSize());
  u_.resize(x_.totalSize());
  for(const auto c : calculators_)
  {
    auto mr = c->m() - c->r();
    StB_.emplace_back(mr, y_.totalSize());
    StZ_.emplace_back(mr, z_.totalSize());
    Stc_.emplace_back(mr);
  }
  firstY_.resize(y_.variables().size(), true);
  firstZ_.resize(z_.variables().size(), true);

  // We could put all matrices to zero, but we don't do it to ease possible debug
  //(at least under msvc, uninitialized values are easy to spot)
  // We thus only initialize to 0 the parts of B and Z that will be used
  int m = 0;
  for(size_t k = 0; k < substitutions_.size(); ++k)
  {
    int mk = 0;
    for(auto i : sub2cstr_[k])
    {
      mk += constraints_[i]->size();
    }
    for(auto i : SYdependencies_[k])
    {
      auto ry = y_[i]->getMappingIn(y_);
      B_.block(m, ry.start, mk, ry.dim).setZero();
    }
    for(auto i : SZdependencies_[k])
    {
      auto rz = z_[i]->getMappingIn(z_);
      Z_.block(m, rz.start, mk, rz.dim).setZero();
    }
    m += mk;
  }
}

void SubstitutionUnit::computeDependencies()
{
  for(size_t k = 0; k < substitutions_.size(); ++k)
  {
    std::set<int> sy;
    std::set<int> sz;
    for(auto i : sub2cstr_[k])
    {
      sy.insert(constraintsY_[i].begin(), constraintsY_[i].end());
      for(auto j : CXdependencies_[i])
      {
        sy.insert(XYdependencies_[j].begin(), XYdependencies_[j].end());
        sz.insert(XZdependencies_[j].begin(), XZdependencies_[j].end());
      }
    }

    SYdependencies_[k].insert(SYdependencies_[k].end(), sy.begin(), sy.end());
    SZdependencies_[k].insert(SZdependencies_[k].end(), sz.begin(), sz.end());

    for(auto i : sub2x_[k])
    {
      XYdependencies_[i] = SYdependencies_[k];
      XZdependencies_[i] = SZdependencies_[k];
      if(z_[static_cast<int>(k)]->size() > 0)
      {
        XZdependencies_[i].push_back(static_cast<int>(k));
      }
    }
  }
}

void SubstitutionUnit::createFunctions()
{
  for(size_t i = 0; i < x_.variables().size(); ++i)
  {
    std::vector<VariablePtr> vars;
    for(auto j : XYdependencies_[i])
    {
      vars.push_back(y_[j]);
    }
    for(auto j : XZdependencies_[i])
    {
      vars.push_back(z_[j]);
    }
    varSubstitutions_.emplace_back(new function::BasicLinearFunction(x_[static_cast<int>(i)]->size(), vars));
  }

  for(size_t k = 0; k < substitutions_.size(); ++k)
  {
    std::vector<VariablePtr> vars;
    for(auto j : SYdependencies_[k])
    {
      vars.push_back(y_[j]);
    }
    for(auto j : SZdependencies_[k])
    {
      vars.push_back(z_[j]);
    }
    remaining_.emplace_back(
        new constraint::BasicLinearConstraint(static_cast<int>(calculators_[k]->m() - calculators_[k]->r()), vars,
                                              constraint::Type::EQUAL, constraint::RHS::AS_GIVEN));
  }
}

} // namespace internal
} // namespace hint
} // namespace tvm
