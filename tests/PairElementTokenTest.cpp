/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#include <tvm/internal/PairElementToken.h>

#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

using namespace tvm::internal;


TEST_CASE("Lifetime 1")
{
  std::unique_ptr<PairElementToken>(t3);
  {
    PairElementToken t1;
    {
      // Create t2 from t1
      PairElementToken t2;
      t1.pairWith(t2);
      FAST_CHECK_UNARY(t1.isPaired());
      FAST_CHECK_UNARY(t2.isPaired());
      FAST_CHECK_UNARY(t1.isPairedWith(t2));
      FAST_CHECK_UNARY(t2.isPairedWith(t1));
      FAST_CHECK_EQ(&t1.otherPairElement(), &t2);
      FAST_CHECK_EQ(&t2.otherPairElement(), &t1);
      // End of t2's life
    }
    FAST_CHECK_UNARY(!t1.isPaired());
    // Create t3 from t1
    t3.reset(new PairElementToken());
    t3->pairWith(t1);
    FAST_CHECK_UNARY(t1.isPaired());
    FAST_CHECK_UNARY(t3->isPaired());
    FAST_CHECK_UNARY(t1.isPairedWith(*t3));
    FAST_CHECK_UNARY(t3->isPairedWith(t1));
    FAST_CHECK_EQ(&t1.otherPairElement(), t3.get());
    FAST_CHECK_EQ(&t3->otherPairElement(), &t1);
    // End of t1's life
  }
  FAST_CHECK_UNARY(!t3->isPaired());
  // End of t3's life
}


std::pair<PairElementToken, PairElementToken> createPair1()
{
  PairElementToken t1;
  PairElementToken t2;
  t1.pairWith(t2);
  return { std::move(t1), std::move(t2) };
}

TEST_CASE("Lifetime 2")
{
  auto [t1, t2] = createPair1();
  FAST_CHECK_UNARY(t1.isPaired());
  FAST_CHECK_UNARY(t2.isPaired());
  FAST_CHECK_UNARY(t1.isPairedWith(t2));
  FAST_CHECK_UNARY(t2.isPairedWith(t1));
  FAST_CHECK_EQ(&t1.otherPairElement(), &t2);
  FAST_CHECK_EQ(&t2.otherPairElement(), &t1);
}


std::pair<PairElementToken, PairElementToken> createPair2()
{
  std::pair<PairElementToken, PairElementToken> p;
  p.first.pairWith(p.second);
  return p;
}

TEST_CASE("Lifetime 3")
{
  PairElementToken t1;
  {
    PairElementToken t2;
    std::tie(t1, t2) = createPair2();
    FAST_CHECK_UNARY(t1.isPaired());
    FAST_CHECK_UNARY(t2.isPaired());
    FAST_CHECK_UNARY(t1.isPairedWith(t2));
    FAST_CHECK_UNARY(t2.isPairedWith(t1));
    FAST_CHECK_EQ(&t1.otherPairElement(), &t2);
    FAST_CHECK_EQ(&t2.otherPairElement(), &t1);
  }
  FAST_CHECK_UNARY(!t1.isPaired());
}


PairElementTokenHandle addToken(std::vector<PairElementToken>& tokens)
{
  tokens.emplace_back();
  return PairElementTokenHandle(tokens.back());
}

TEST_CASE("Use with vector and handle")
{
  std::vector<PairElementToken> tokens1;
  {
    std::vector<PairElementToken> tokens2;

    //adding paired tokens in tokens1 and tokens2 with various methods
    tokens2.emplace_back(addToken(tokens1));
    tokens2.push_back(addToken(tokens1));
    auto h = addToken(tokens1);
    tokens2.emplace_back(std::move(h));
    FAST_CHECK_UNARY(tokens1[0].isPairedWith(tokens2[0]));
    FAST_CHECK_UNARY(tokens2[0].isPairedWith(tokens1[0]));
    FAST_CHECK_UNARY(tokens1[1].isPairedWith(tokens2[1]));
    FAST_CHECK_UNARY(tokens2[1].isPairedWith(tokens1[1]));
    FAST_CHECK_UNARY(tokens1[2].isPairedWith(tokens2[2]));
    FAST_CHECK_UNARY(tokens2[2].isPairedWith(tokens1[2]));

    //removing an element
    tokens1.erase(tokens1.begin());
    FAST_CHECK_UNARY(!tokens2[0].isPaired());
    FAST_CHECK_UNARY(tokens1[0].isPairedWith(tokens2[1]));
    FAST_CHECK_UNARY(tokens2[1].isPairedWith(tokens1[0]));
    FAST_CHECK_UNARY(tokens1[1].isPairedWith(tokens2[2]));
    FAST_CHECK_UNARY(tokens2[2].isPairedWith(tokens1[1]));
  }
  
  // tokens2 does not exist anymore
  FAST_CHECK_UNARY(!tokens1[0].isPaired());
  FAST_CHECK_UNARY(!tokens1[1].isPaired());
}