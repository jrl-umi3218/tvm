#include <assert.h>
#include "MatrixProperties.h"

namespace
{
  using namespace tvm;

  /** Partial comparison between positivness caracteristics.
    * We have the following implications (from left to right)
    *
    *      psd
    *    /     \
    * pd        \
    *    \       \
    *      unz --- u --- NA
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
    *  - unz = non zero undefinite
    *  - u   = undefinite
    *  - na  = non available
    *
    * For a and b two positivness caracteristics, a > b if a a implies b.
    * If a and b are unrelated (e.g. pd and nsd), an assertion is fired.
    */
  bool greaterThan(Positivness a, Positivness b)
  {
    // The following table translates the above graph:
    //  - table[i][j] = 1 if i > j
    //  - table[i][j] = -1 if i <= j
    //  - table[i][j] = 0 if the comparison is not valid.
                            // NA  | PSD | PD  | NSD | ND  |  U  | UNZ  
    const int table[7][7] = {
                  /* NA  */  { -1  , -1  , -1  , -1  , -1  , -1  , -1  },
                  /* PSD */  {  1  , -1  , -1  ,  0  ,  0  ,  1  ,  0  },
                  /* PD  */  {  1  ,  1  , -1  ,  0  ,  0  ,  1  ,  1  },
                  /* NSD */  {  1  ,  0  ,  0  ,  1  , -1  ,  1  ,  1  },
                  /* ND  */  {  1  ,  0  ,  0  ,  1  , -1  ,  1  ,  1  },
                  /*  U  */  {  1  , -1  , -1  , -1  , -1  , -1  , -1  },
                  /* UNZ */  {  1  ,  0  , -1  ,  0  , -1  ,  1  ,  1  }};
    assert(table[static_cast<int>(a)][static_cast<int>(b)] != 0 && "Invalid comparison");

    return table[static_cast<int>(a)][static_cast<int>(b)] > 0;
  }

  Positivness max(Positivness a, Positivness b)
  {
    if (greaterThan(a,b))
      return a;
    else
      return b;
  }

  Positivness promote(Positivness p, bool invertible)
  {
    //possibly promote if the matrix is invertible
    if (invertible)
    {
      switch (p)
      {
      case Positivness::POSITIVE_SEMIDEFINITE: p = Positivness::POSITIVE_DEFINITE;   break;
      case Positivness::NEGATIVE_SEMIDEFINITE: p = Positivness::NEGATIVE_DEFINITE;   break;
      case Positivness::UNDEFINITE:            p = Positivness::NON_ZERO_UNDEFINITE; break;
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

  bool deduceSymmetry(MatrixShape shape, Positivness positivness)
  {
    return shape > MatrixShape::GENERAL || positivness != Positivness::NA;
  }

  bool deduceInvertibility(MatrixShape shape, Positivness positivness)
  {
    return shape == MatrixShape::IDENTITY
        || shape == MatrixShape::MINUS_IDENTITY
        || positivness == Positivness::POSITIVE_DEFINITE
        || positivness == Positivness::NEGATIVE_DEFINITE
        || positivness == Positivness::NON_ZERO_UNDEFINITE;
  }

  Positivness deducePositivness(MatrixShape shape, Positivness positivness, bool invertible)
  {
    Positivness p;
    //get default positivness for the shape
    switch (shape)
    {
    case MatrixShape::GENERAL:              p = Positivness::NA;                break;
    case MatrixShape::DIAGONAL:             p = Positivness::UNDEFINITE;        break;
    case MatrixShape::MULTIPLE_OF_IDENTITY: p = Positivness::UNDEFINITE;        break;
    case MatrixShape::IDENTITY:             p = Positivness::POSITIVE_DEFINITE; break;
    case MatrixShape::MINUS_IDENTITY:       p = Positivness::NEGATIVE_DEFINITE; break;
    case MatrixShape::ZERO:                 p = Positivness::UNDEFINITE;        break;
    }

    //get tighter positiveness between default one and user-provided one, possibly promoted if invertible
    return max(promote(p, invertible), promote(positivness, invertible));
  }
}

namespace tvm
{
  MatrixProperties::MatrixProperties(MatrixShape shape, Positivness positivness)
    : MatrixProperties(deduceConstance(shape), shape, positivness)
  {
  }

  MatrixProperties::MatrixProperties(bool constant, MatrixShape shape, Positivness positivness)
    : MatrixProperties(deduceInvertibility(shape, positivness), constant, shape, positivness)
  {
    assert((constant || !deduceConstance(shape)) && "You marked as non constant a matrix that is necessarily constant");
  }

  MatrixProperties::MatrixProperties(bool invertible, bool constant, MatrixShape shape, Positivness positivness)
    : constant_(constant || deduceConstance(shape))
    , invertible_(invertible || deduceInvertibility(shape,positivness))
    , shape_(shape)
    , symmetric_(deduceSymmetry(shape, positivness))
    , positivness_(deducePositivness(shape,positivness,invertible))
  {
    assert((constant || !deduceConstance(shape)) && "You marked as non constant a matrix that is necessarily constant.");
    assert((invertible || !deduceInvertibility(shape, positivness)) && "You marked as non-invertible a matrix that is necessarily invertible.");
    assert(!(invertible && shape == MatrixShape::ZERO));
    assert(!(shape == MatrixShape::ZERO && positivness == Positivness::POSITIVE_DEFINITE));
    assert(!(shape == MatrixShape::ZERO && positivness == Positivness::NEGATIVE_DEFINITE));
    assert(!(shape == MatrixShape::ZERO && positivness == Positivness::NON_ZERO_UNDEFINITE));
    assert(!(shape == MatrixShape::IDENTITY && positivness == Positivness::NEGATIVE_SEMIDEFINITE));
    assert(!(shape == MatrixShape::IDENTITY && positivness == Positivness::NEGATIVE_DEFINITE));
    assert(!(shape == MatrixShape::MINUS_IDENTITY && positivness == Positivness::POSITIVE_SEMIDEFINITE));
    assert(!(shape == MatrixShape::MINUS_IDENTITY && positivness == Positivness::POSITIVE_DEFINITE));
  }

}