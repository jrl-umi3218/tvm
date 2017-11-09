#include <assert.h>
#include <stdexcept>


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
  bool greaterThan(MatrixProperties::Positiveness a, MatrixProperties::Positiveness b)
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
    if (table[static_cast<int>(a)][static_cast<int>(b)] == 0)
      throw std::runtime_error("Invalid comparison");

    return table[static_cast<int>(a)][static_cast<int>(b)] > 0;
  }

  MatrixProperties::Positiveness max(MatrixProperties::Positiveness a, MatrixProperties::Positiveness b)
  {
    if (greaterThan(a,b))
      return a;
    else
      return b;
  }

  MatrixProperties::Positiveness promote(MatrixProperties::Positiveness p, bool invertible)
  {
    //possibly promote if the matrix is invertible
    if (invertible)
    {
      switch (p)
      {
      case MatrixProperties::Positiveness::POSITIVE_SEMIDEFINITE: p = MatrixProperties::Positiveness::POSITIVE_DEFINITE;   break;
      case MatrixProperties::Positiveness::NEGATIVE_SEMIDEFINITE: p = MatrixProperties::Positiveness::NEGATIVE_DEFINITE;   break;
      case MatrixProperties::Positiveness::INDEFINITE:            p = MatrixProperties::Positiveness::NON_ZERO_INDEFINITE; break;
      default: break;
      }
    }
    return p;
  }

  bool deduceConstance(MatrixProperties::Shape shape)
  {
    return shape == MatrixProperties::Shape::IDENTITY
        || shape == MatrixProperties::Shape::MINUS_IDENTITY
        || shape == MatrixProperties::Shape::ZERO;
  }

  bool deduceSymmetry(MatrixProperties::Shape shape, MatrixProperties::Positiveness positiveness)
  {
    return shape >= MatrixProperties::Shape::DIAGONAL || positiveness != MatrixProperties::Positiveness::NA;
  }

  bool deduceInvertibility(MatrixProperties::Shape shape, MatrixProperties::Positiveness positiveness)
  {
    return shape == MatrixProperties::Shape::IDENTITY
        || shape == MatrixProperties::Shape::MINUS_IDENTITY
        || positiveness == MatrixProperties::Positiveness::POSITIVE_DEFINITE
        || positiveness == MatrixProperties::Positiveness::NEGATIVE_DEFINITE
        || positiveness == MatrixProperties::Positiveness::NON_ZERO_INDEFINITE;
  }

  MatrixProperties::Shape deduceShape(MatrixProperties::Shape shape, MatrixProperties::Positiveness positiveness)
  {
    //if triangular and symmetric we deduce diagonal
    if ((shape == MatrixProperties::Shape::LOWER_TRIANGULAR || shape == MatrixProperties::Shape::UPPER_TRIANGULAR)
      && positiveness != MatrixProperties::Positiveness::NA)
    {
      return MatrixProperties::Shape::DIAGONAL;
    }
    else
      return shape;
  }

  MatrixProperties::Positiveness deducePositiveness(MatrixProperties::Shape shape, MatrixProperties::Positiveness positiveness, bool invertible)
  {
    MatrixProperties::Positiveness p = MatrixProperties::Positiveness::NA;
    //get default positiveness for the shape
    switch (shape)
    {
    case MatrixProperties::Shape::DIAGONAL:             p = MatrixProperties::Positiveness::INDEFINITE;        break;
    case MatrixProperties::Shape::MULTIPLE_OF_IDENTITY: p = MatrixProperties::Positiveness::INDEFINITE;        break;
    case MatrixProperties::Shape::IDENTITY:             p = MatrixProperties::Positiveness::POSITIVE_DEFINITE; break;
    case MatrixProperties::Shape::MINUS_IDENTITY:       p = MatrixProperties::Positiveness::NEGATIVE_DEFINITE; break;
    case MatrixProperties::Shape::ZERO:                 p = MatrixProperties::Positiveness::INDEFINITE;        break;
    default: break;
    }

    //get tighter positiveness between default one and user-provided one, possibly promoted if invertible
    return max(promote(p, invertible), promote(positiveness, invertible));
  }
}

namespace tvm
{
  MatrixProperties::MatrixProperties()
    : constant_(false)
    , invertible_(false)
    , shape_(MatrixProperties::GENERAL)
    , symmetric_(false)
    , positiveness_(MatrixProperties::NA)
  {
  }


  void MatrixProperties::build(const MatrixProperties::Arguments& args, const std::pair<bool, bool>& checks)
  {
    if ((args.shape == Shape::ZERO && args.positiveness == Positiveness::POSITIVE_DEFINITE)
      || (args.shape == Shape::ZERO && args.positiveness == Positiveness::NEGATIVE_DEFINITE)
      || (args.shape == Shape::ZERO && args.positiveness == Positiveness::NON_ZERO_INDEFINITE)
      || (args.shape == Shape::IDENTITY && args.positiveness == Positiveness::NEGATIVE_SEMIDEFINITE)
      || (args.shape == Shape::IDENTITY && args.positiveness == Positiveness::NEGATIVE_DEFINITE)
      || (args.shape == Shape::MINUS_IDENTITY && args.positiveness == Positiveness::POSITIVE_SEMIDEFINITE)
      || (args.shape == Shape::MINUS_IDENTITY && args.positiveness == Positiveness::POSITIVE_DEFINITE))
    {
      throw std::runtime_error("Incompatible shape and positiveness properties.");
    }

    if (checks.first && !args.constant && deduceConstance(args.shape))
      throw std::runtime_error("You marked as non constant a matrix that is necessarily constant.");

    if (checks.second)
    {
      if (!args.invertible && deduceInvertibility(args.shape, args.positiveness))
        throw std::runtime_error("You marked as non-invertible a matrix that is necessarily invertible.");
      if (args.invertible && args.shape == Shape::ZERO)
        throw std::runtime_error("You marked as invertible the matrix 0.");
    }

    constant_ = args.constant || deduceConstance(args.shape);
    invertible_ = args.invertible || deduceInvertibility(args.shape, args.positiveness);
    shape_ = deduceShape(args.shape, args.positiveness);
    symmetric_ = deduceSymmetry(args.shape, args.positiveness);
    positiveness_ = deducePositiveness(args.shape, args.positiveness, args.invertible);
  }

}
