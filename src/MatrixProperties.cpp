#include <assert.h>
#include "MatrixProperties.h"

namespace
{
  using namespace tvm;

  /** Partial comparison between positiveness caracteristics.
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
    * For a and b two positiveness caracteristics, a > b if a a implies b.
    * If a and b are unrelated (e.g. pd and nsd), an assertion is fired.
    */
  bool greaterThan(Positiveness a, Positiveness b)
  {
    // The following table translates the above graph:
    //  - table[i][j] = 1 if i > j
    //  - table[i][j] = -1 if i <= j
    //  - table[i][j] = 0 if the comparison is not valid.
                            // NA  | PSD | PD  | NSD | ND  |  I  | INZ  
    const int table[7][7] = {
                  /* NA  */  { -1  , -1  , -1  , -1  , -1  , -1  , -1  },
                  /* PSD */  {  1  , -1  , -1  ,  0  ,  0  ,  1  ,  0  },
                  /* PD  */  {  1  ,  1  , -1  ,  0  ,  0  ,  1  ,  1  },
                  /* NSD */  {  1  ,  0  ,  0  ,  1  , -1  ,  1  ,  1  },
                  /* ND  */  {  1  ,  0  ,  0  ,  1  , -1  ,  1  ,  1  },
                  /*  I  */  {  1  , -1  , -1  , -1  , -1  , -1  , -1  },
                  /* INZ */  {  1  ,  0  , -1  ,  0  , -1  ,  1  ,  1  }};
    assert(table[static_cast<int>(a)][static_cast<int>(b)] != 0 && "Invalid comparison");

    return table[static_cast<int>(a)][static_cast<int>(b)] > 0;
  }

  Positiveness max(Positiveness a, Positiveness b)
  {
    if (greaterThan(a,b))
      return a;
    else
      return b;
  }

  Positiveness promote(Positiveness p, bool invertible)
  {
    //possibly promote if the matrix is invertible
    if (invertible)
    {
      switch (p)
      {
      case Positiveness::POSITIVE_SEMIDEFINITE: p = Positiveness::POSITIVE_DEFINITE;   break;
      case Positiveness::NEGATIVE_SEMIDEFINITE: p = Positiveness::NEGATIVE_DEFINITE;   break;
      case Positiveness::INDEFINITE:            p = Positiveness::NON_ZERO_INDEFINITE; break;
      default: break;
      }
    }
    return p;
  }

  bool deduceConstance(MatrixShape shape)
  {
    return shape == MatrixShape::IDENTITY 
        || shape == MatrixShape::MINUS_IDENTITY
        || shape == MatrixShape::ZERO;
  }

  bool deduceSymmetry(MatrixShape shape, Positiveness positiveness)
  {
    return shape >= MatrixShape::DIAGONAL || positiveness != Positiveness::NA;
  }

  bool deduceInvertibility(MatrixShape shape, Positiveness positiveness)
  {
    return shape == MatrixShape::IDENTITY
        || shape == MatrixShape::MINUS_IDENTITY
        || positiveness == Positiveness::POSITIVE_DEFINITE
        || positiveness == Positiveness::NEGATIVE_DEFINITE
        || positiveness == Positiveness::NON_ZERO_INDEFINITE;
  }

  MatrixShape deduceShape(MatrixShape shape, Positiveness positiveness)
  {
    //if triangular and symmetric we deduce diagonal
    if ((shape == MatrixShape::LOWER_TRIANGULAR || shape == MatrixShape::UPPER_TRIANGULAR)
      && positiveness != Positiveness::NA)
    {
      return MatrixShape::DIAGONAL;
    }
    else
      return shape;
  }

  Positiveness deducePositiveness(MatrixShape shape, Positiveness positiveness, bool invertible)
  {
    Positiveness p;
    //get default positiveness for the shape
    switch (shape)
    {
    case MatrixShape::GENERAL:              p = Positiveness::NA;                break;
    case MatrixShape::LOWER_TRIANGULAR:     p = Positiveness::NA;                break;
    case MatrixShape::UPPER_TRIANGULAR:     p = Positiveness::NA;                break;
    case MatrixShape::DIAGONAL:             p = Positiveness::INDEFINITE;        break;
    case MatrixShape::MULTIPLE_OF_IDENTITY: p = Positiveness::INDEFINITE;        break;
    case MatrixShape::IDENTITY:             p = Positiveness::POSITIVE_DEFINITE; break;
    case MatrixShape::MINUS_IDENTITY:       p = Positiveness::NEGATIVE_DEFINITE; break;
    case MatrixShape::ZERO:                 p = Positiveness::INDEFINITE;        break;
    }

    //get tighter positiveness between default one and user-provided one, possibly promoted if invertible
    return max(promote(p, invertible), promote(positiveness, invertible));
  }
}

namespace tvm
{
  MatrixProperties::MatrixProperties(MatrixShape shape, Positiveness positiveness)
    : MatrixProperties(deduceConstance(shape), shape, positiveness)
  {
  }

  MatrixProperties::MatrixProperties(bool constant, MatrixShape shape, Positiveness positiveness)
    : MatrixProperties(deduceInvertibility(shape, positiveness), constant, shape, positiveness)
  {
    assert((constant || !deduceConstance(shape)) && "You marked as non constant a matrix that is necessarily constant");
  }

  MatrixProperties::MatrixProperties(bool invertible, bool constant, MatrixShape shape, Positiveness positiveness)
    : constant_(constant || deduceConstance(shape))
    , invertible_(invertible || deduceInvertibility(shape,positiveness))
    , shape_(deduceShape(shape,positiveness))
    , symmetric_(deduceSymmetry(shape, positiveness))
    , positiveness_(deducePositiveness(shape,positiveness,invertible))
  {
    assert((constant || !deduceConstance(shape)) && "You marked as non constant a matrix that is necessarily constant.");
    assert((invertible || !deduceInvertibility(shape, positiveness)) && "You marked as non-invertible a matrix that is necessarily invertible.");
    assert(!(invertible && shape == MatrixShape::ZERO));
    assert(!(shape == MatrixShape::ZERO && positiveness == Positiveness::POSITIVE_DEFINITE));
    assert(!(shape == MatrixShape::ZERO && positiveness == Positiveness::NEGATIVE_DEFINITE));
    assert(!(shape == MatrixShape::ZERO && positiveness == Positiveness::NON_ZERO_INDEFINITE));
    assert(!(shape == MatrixShape::IDENTITY && positiveness == Positiveness::NEGATIVE_SEMIDEFINITE));
    assert(!(shape == MatrixShape::IDENTITY && positiveness == Positiveness::NEGATIVE_DEFINITE));
    assert(!(shape == MatrixShape::MINUS_IDENTITY && positiveness == Positiveness::POSITIVE_SEMIDEFINITE));
    assert(!(shape == MatrixShape::MINUS_IDENTITY && positiveness == Positiveness::POSITIVE_DEFINITE));
  }

}