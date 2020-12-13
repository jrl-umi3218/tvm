/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/internal/MatrixProperties.h>

#include <algorithm>
#include <assert.h>
#include <stdexcept>

using namespace tvm::internal;
using Constness = MatrixProperties::Constness;
using Invertibility = MatrixProperties::Invertibility;
using Positiveness = MatrixProperties::Positiveness;
using Shape = MatrixProperties::Shape;

namespace
{
/** Partial comparison between positiveness characteristics.
 * We have the following implications (from left to right)
 *
 *      psd
 *    /     \
 * pd        \
 *    \       \
 *      inz --- i --- NA
 *    /       /
 * nd        /
 *    \     /
 *      nsd
 *
 * with
 *  - pd  = positive definite
 *  - psd = positive semidefinite
 *  - nd  = negative definite
 *  - nsd = negative semidefinite
 *  - inz = non zero indefinite
 *  - i   = indefinite
 *  - na  = non available
 *
 * For a and b two positiveness characteristics, a > b if a implies b.
 * If a and b are unrelated (e.g. pd and nsd), an error is thrown.
 */
bool greaterThan(Positiveness a, Positiveness b)
{
  // The following table translates the above graph:
  //  - table[i][j] = 1 if i > j
  //  - table[i][j] = -1 if i <= j
  //  - table[i][j] = 0 if the comparison is not valid.
  // NA  | PSD | PD  | NSD | ND  |  I  | INZ
  const int table[7][7] = {/* NA  */ {-1, -1, -1, -1, -1, -1, -1},
                           /* PSD */ {1, -1, -1, 0, 0, 1, 0},
                           /* PD  */ {1, 1, -1, 0, 0, 1, 1},
                           /* NSD */ {1, 0, 0, 1, -1, 1, 1},
                           /* ND  */ {1, 0, 0, 1, -1, 1, 1},
                           /*  I  */ {1, -1, -1, -1, -1, -1, -1},
                           /* INZ */ {1, 0, -1, 0, -1, 1, 1}};
  if(table[static_cast<int>(a)][static_cast<int>(b)] == 0)
    throw std::runtime_error("Invalid comparison");

  return table[static_cast<int>(a)][static_cast<int>(b)] > 0;
}

Positiveness max(Positiveness a, Positiveness b)
{
  if(greaterThan(a, b))
    return a;
  else
    return b;
}

Positiveness promote(Positiveness p, bool invertible)
{
  // possibly promote if the matrix is invertible
  if(invertible)
  {
    switch(p)
    {
      case Positiveness::POSITIVE_SEMIDEFINITE:
        p = Positiveness::POSITIVE_DEFINITE;
        break;
      case Positiveness::NEGATIVE_SEMIDEFINITE:
        p = Positiveness::NEGATIVE_DEFINITE;
        break;
      case Positiveness::INDEFINITE:
        p = Positiveness::NON_ZERO_INDEFINITE;
        break;
      default:
        break;
    }
  }
  return p;
}

bool deduceConstance(Shape shape)
{
  return shape == Shape::IDENTITY || shape == Shape::MINUS_IDENTITY || shape == Shape::ZERO;
}

bool deduceSymmetry(Shape shape, Positiveness positiveness)
{
  return shape >= Shape::DIAGONAL || positiveness != Positiveness::NA;
}

bool deduceInvertibility(Shape shape, Positiveness positiveness)
{
  return shape == Shape::IDENTITY || shape == Shape::MINUS_IDENTITY || positiveness == Positiveness::POSITIVE_DEFINITE
         || positiveness == Positiveness::NEGATIVE_DEFINITE || positiveness == Positiveness::NON_ZERO_INDEFINITE;
}

Shape deduceShape(Shape shape, Positiveness positiveness)
{
  // if triangular and symmetric we deduce diagonal
  if((shape == Shape::LOWER_TRIANGULAR || shape == Shape::UPPER_TRIANGULAR) && positiveness != Positiveness::NA)
  {
    return Shape::DIAGONAL;
  }
  else
    return shape;
}

Positiveness deducePositiveness(Shape shape, Positiveness positiveness, bool invertible)
{
  Positiveness p = Positiveness::NA;
  // get default positiveness for the shape
  switch(shape)
  {
    case Shape::DIAGONAL:
      p = Positiveness::INDEFINITE;
      break;
    case Shape::MULTIPLE_OF_IDENTITY:
      p = Positiveness::INDEFINITE;
      break;
    case Shape::IDENTITY:
      p = Positiveness::POSITIVE_DEFINITE;
      break;
    case Shape::MINUS_IDENTITY:
      p = Positiveness::NEGATIVE_DEFINITE;
      break;
    case Shape::ZERO:
      p = Positiveness::INDEFINITE;
      break;
    default:
      break;
  }

  // get tighter positiveness between default one and user-provided one, possibly promoted if invertible
  return max(promote(p, invertible), promote(positiveness, invertible));
}

} // namespace

namespace tvm
{

namespace internal
{

MatrixProperties::MatrixProperties()
: constant_(false), invertible_(false), shape_(MatrixProperties::GENERAL), symmetric_(false),
  positiveness_(MatrixProperties::NA)
{}

void MatrixProperties::build(const MatrixProperties::Arguments & args, const std::pair<bool, bool> & checks)
{
  if((args.shape == Shape::ZERO && args.positiveness == Positiveness::POSITIVE_DEFINITE)
     || (args.shape == Shape::ZERO && args.positiveness == Positiveness::NEGATIVE_DEFINITE)
     || (args.shape == Shape::ZERO && args.positiveness == Positiveness::NON_ZERO_INDEFINITE)
     || (args.shape == Shape::IDENTITY && args.positiveness == Positiveness::NEGATIVE_SEMIDEFINITE)
     || (args.shape == Shape::IDENTITY && args.positiveness == Positiveness::NEGATIVE_DEFINITE)
     || (args.shape == Shape::MINUS_IDENTITY && args.positiveness == Positiveness::POSITIVE_SEMIDEFINITE)
     || (args.shape == Shape::MINUS_IDENTITY && args.positiveness == Positiveness::POSITIVE_DEFINITE))
  {
    throw std::runtime_error("Incompatible shape and positiveness properties.");
  }

  if(checks.first && !args.constant && deduceConstance(args.shape))
    throw std::runtime_error("You marked as non constant a matrix that is necessarily constant.");

  if(checks.second)
  {
    if(!args.invertible && deduceInvertibility(args.shape, args.positiveness))
      throw std::runtime_error("You marked as non-invertible a matrix that is necessarily invertible.");
    if(args.invertible && args.shape == Shape::ZERO)
      throw std::runtime_error("You marked as invertible the matrix 0.");
  }

  constant_ = args.constant || deduceConstance(args.shape);
  invertible_ = args.invertible || deduceInvertibility(args.shape, args.positiveness);
  shape_ = deduceShape(args.shape, args.positiveness);
  symmetric_ = deduceSymmetry(args.shape, args.positiveness);
  positiveness_ = deducePositiveness(args.shape, args.positiveness, args.invertible);
}

Constness operator-(const Constness & c) { return c; }
Constness operator*(double, const Constness & c) { return c; }
Constness operator+(const Constness & c1, const Constness & c2) { return c1 && c2; }
Constness operator-(const Constness & c1, const Constness & c2) { return c1 && c2; }
Constness operator*(const Constness & c1, const Constness & c2) { return c1 && c2; }
Invertibility operator-(const Invertibility & i) { return i; }
Invertibility operator*(double d, const Invertibility & i) { return d == 0 ? Invertibility(false) : i; }
Invertibility operator+(const Invertibility &, const Invertibility &) { return false; }
Invertibility operator-(const Invertibility &, const Invertibility &) { return false; }
Invertibility operator*(const Invertibility & i1, const Invertibility & i2) { return i1 && i2; }

Positiveness operator-(const Positiveness & p)
{
  switch(p)
  {
    case Positiveness::NA:
      return Positiveness::NA;
    case Positiveness::POSITIVE_SEMIDEFINITE:
      return Positiveness::NEGATIVE_SEMIDEFINITE;
    case Positiveness::POSITIVE_DEFINITE:
      return Positiveness::NEGATIVE_DEFINITE;
    case Positiveness::NEGATIVE_SEMIDEFINITE:
      return Positiveness::POSITIVE_SEMIDEFINITE;
    case Positiveness::NEGATIVE_DEFINITE:
      return Positiveness::POSITIVE_DEFINITE;
    case Positiveness::INDEFINITE:
      return Positiveness::INDEFINITE;
    case Positiveness::NON_ZERO_INDEFINITE:
      return Positiveness::NON_ZERO_INDEFINITE;
    default:
      throw std::runtime_error("Case not possible.");
  }
}

Positiveness operator*(double d, const Positiveness & p)
{
  if(d < 0)
    return -p;
  else if(d > 0)
    return p;
  else if(p == Positiveness::NA)
    return p; // matrix is zero but maybe not square
  else
    return Positiveness::INDEFINITE; // matrix is zero and square
}

Positiveness operator+(const Positiveness & p1, const Positiveness & p2)
{
  if(p1 == Positiveness::NA || p2 == Positiveness::NA)
    return Positiveness::NA;
  else if((p1 == Positiveness::POSITIVE_SEMIDEFINITE || p1 == Positiveness::POSITIVE_DEFINITE)
          && (p2 == Positiveness::POSITIVE_SEMIDEFINITE || p2 == Positiveness::POSITIVE_DEFINITE))
    return max(p1, p2);
  else if((p1 == Positiveness::NEGATIVE_SEMIDEFINITE || p1 == Positiveness::NEGATIVE_DEFINITE)
          && (p2 == Positiveness::NEGATIVE_SEMIDEFINITE || p2 == Positiveness::NEGATIVE_DEFINITE))
    return max(p1, p2);
  else
    return Positiveness::INDEFINITE;
}

Positiveness operator-(const Positiveness & p1, const Positiveness & p2) { return p1 + (-p2); }

Positiveness operator*(const Positiveness &, const Positiveness &)
{
  // The product of two symmetric matrices is in not symmetric (unless they permute).
  return Positiveness::NA;
}

Shape operator-(const Shape & s)
{
  if(s == Shape::IDENTITY)
    return Shape::MINUS_IDENTITY;
  else if(s == Shape::MINUS_IDENTITY)
    return Shape::IDENTITY;
  else
    return s;
}

Shape operator*(double d, const Shape & s)
{
  if(d < 0)
    return -s;
  else if(d > 0)
    return s;
  else
    return Shape::ZERO;
}

Shape operator+(const Shape & s1, const Shape & s2)
{
  // the addition is commutative so we order s1 and s2 to consider only about half the cases
  auto s = std::minmax(s1, s2);

  //    |  G |  L |  U |  D | aI |  I | -I |  0 |
  //  G |  G |  G |  G |  G |  G |  G |  G |  G |
  //  L |    |  L |  G |  L |  L |  L |  L |  L |
  //  U |    |    |  U |  U |  U |  U |  U |  U |
  //  D |    |    |    |  D |  D |  D |  D |  D |
  // aI |    |    |    |    | aI | aI | aI | aI |
  //  I |    |    |    |    |    | aI |  0 |  I |
  // -I |    |    |    |    |    |    | aI | -I |
  //  0 |    |    |    |    |    |    |    |  0 |
  switch(s.first)
  {
    case Shape::GENERAL:
      return Shape::GENERAL;
    case Shape::LOWER_TRIANGULAR:
      return s.second == Shape::UPPER_TRIANGULAR ? Shape::GENERAL : Shape::LOWER_TRIANGULAR;
    case Shape::UPPER_TRIANGULAR:
      return Shape::UPPER_TRIANGULAR;
    case Shape::DIAGONAL:
      return Shape::DIAGONAL;
    case Shape::MULTIPLE_OF_IDENTITY:
      return Shape::MULTIPLE_OF_IDENTITY;
    case Shape::IDENTITY:
      if(s.second == Shape::IDENTITY)
        return Shape::MULTIPLE_OF_IDENTITY;
      else if(s.second == Shape::MINUS_IDENTITY)
        return Shape::ZERO;
      else
        return Shape::IDENTITY;
    case Shape::MINUS_IDENTITY:
      return s.second == Shape::MINUS_IDENTITY ? Shape::MULTIPLE_OF_IDENTITY : Shape::MINUS_IDENTITY;
    case Shape::ZERO:
      return Shape::ZERO;
    default:
      throw std::runtime_error("Case not possible.");
  }
}

Shape operator-(const Shape & s1, const Shape & s2) { return s1 + (-s2); }

Shape operator*(const Shape & s1, const Shape & s2)
{
  // the multiplication is commutative so we order s1 and s2 to consider only about half the cases
  auto s = std::minmax(s1, s2);

  //    |  G |  L |  U |  D | aI |  I | -I |  0 |
  //  G |  G |  G |  G |  G |  G |  G |  G |  0 |
  //  L |    |  L |  G |  L |  L |  L |  L |  0 |
  //  U |    |    |  U |  U |  U |  U |  U |  0 |
  //  D |    |    |    |  D |  D |  D |  D |  0 |
  // aI |    |    |    |    | aI | aI | aI |  0 |
  //  I |    |    |    |    |    |  I | -I |  0 |
  // -I |    |    |    |    |    |    |  I |  0 |
  //  0 |    |    |    |    |    |    |    |  0 |
  if(s.second == Shape::ZERO)
    return Shape::ZERO;
  switch(s.first)
  {
    case Shape::GENERAL:
      return Shape::GENERAL;
    case Shape::LOWER_TRIANGULAR:
      return s.second == Shape::UPPER_TRIANGULAR ? Shape::GENERAL : Shape::LOWER_TRIANGULAR;
    case Shape::UPPER_TRIANGULAR:
      return Shape::UPPER_TRIANGULAR;
    case Shape::DIAGONAL:
      return Shape::DIAGONAL;
    case Shape::MULTIPLE_OF_IDENTITY:
      return Shape::MULTIPLE_OF_IDENTITY;
    case Shape::IDENTITY:
      return s.second == Shape::IDENTITY ? Shape::IDENTITY : Shape::MINUS_IDENTITY;
    case Shape::MINUS_IDENTITY:
      return Shape::IDENTITY;
    default:
      throw std::runtime_error("Case not possible.");
  }
}

MatrixProperties operator-(const MatrixProperties & p)
{
  return {-p.shape(), -p.positiveness(), -p.constness(), -p.invertibility()};
}

MatrixProperties operator*(double d, const MatrixProperties & p)
{
  return {d * p.shape(), d * p.positiveness(), d * p.constness(), d * p.invertibility()};
}

MatrixProperties operator+(const MatrixProperties & p1, const MatrixProperties & p2)
{
  return {p1.shape() + p2.shape(), p1.positiveness() + p2.positiveness(), p1.constness() + p2.constness(),
          p1.invertibility() + p2.invertibility()};
}

MatrixProperties operator-(const MatrixProperties & p1, const MatrixProperties & p2)
{
  return {p1.shape() - p2.shape(), p1.positiveness() - p2.positiveness(), p1.constness() - p2.constness(),
          p1.invertibility() - p2.invertibility()};
}

MatrixProperties operator*(const MatrixProperties & p1, const MatrixProperties & p2)
{
  return {p1.shape() * p2.shape(), p1.positiveness() * p2.positiveness(), p1.constness() * p2.constness(),
          p1.invertibility() * p2.invertibility()};
}

} // namespace internal

} // namespace tvm
