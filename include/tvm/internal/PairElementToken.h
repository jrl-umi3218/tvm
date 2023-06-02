/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <assert.h>
#include <memory>
#include <stdexcept>

namespace tvm::internal
{
class PairElementToken;

/** Proxy over a PairElementToken from which can be create a paired token.
 *
 * \internal This class allows to delay the construction of the second element
 * a pair of token, avoiding constructions and moves
 */
class PairElementTokenHandle
{
public:
  PairElementTokenHandle() = delete;
  PairElementTokenHandle(const PairElementTokenHandle &) = delete;
  PairElementTokenHandle(PairElementTokenHandle &&) = delete;
  PairElementTokenHandle & operator=(const PairElementTokenHandle &) = delete;
  PairElementTokenHandle & operator=(PairElementTokenHandle &&) = delete;

  explicit PairElementTokenHandle(PairElementToken & t) noexcept;

  bool isValid() const { return token_ != nullptr; }

private:
  void invalidate() { token_ = nullptr; }

  PairElementToken * token_;

  friend PairElementToken;
};

/** A class for representing a token that can be paired with another one.
 * Paired tokens inform each other in case they are moved or destroyed so that
 * they can keep track of one another.
 * Each Token must be unique and therefore cannot be copied.
 */
class PairElementToken
{
public:
  PairElementToken() noexcept {}
  PairElementToken(const PairElementToken &) = delete;             // To keep a pair valid we cannot duplicate tokens
  PairElementToken & operator=(const PairElementToken &) = delete; // idem

  /** Move constructor.*/
  PairElementToken(PairElementToken && other) noexcept : otherPairElement_(other.otherPairElement_)
  {
    other.otherPairElement_ = nullptr; // we need to leave the previous token in an unpaired state.
    if(isPaired())
      otherPairElement_->notifyMove(this); // if paired, we need to notify to otherPairElement_ that this changed place.
  }

  /** Construction from a PairElementTokenHandle. The token constructed and the
   * token pointed to by the handle are paired.
   */
  PairElementToken(PairElementTokenHandle && h) noexcept : otherPairElement_(h.token_)
  {
    assert(h.isValid());
    h.token_->otherPairElement_ = this;
    h.invalidate();
  }

  /** Move-assign operator*/
  PairElementToken & operator=(PairElementToken && other) noexcept
  {
    if(isPaired()) // If we move into an already paired token, we basically erase it and its paired element must be
                   // notified
      otherPairElement_->notifyDeath();
    otherPairElement_ = other.otherPairElement_;
    other.otherPairElement_ = nullptr; // we need to leave the previous token in an unpaired state.
    if(isPaired())
      otherPairElement_->notifyMove(this); // if paired, we need to notify to otherPairElement_ that this changed place.
    return *this;
  }

  /** Move-assign-like operator for PairElementTokenHandle. The token assigned
   * to and the token pointed to by the handle are paired.
   */
  PairElementToken & operator=(PairElementTokenHandle && h) noexcept
  {
    assert(h.isValid());
    if(isPaired()) // If we move into an already paired token, we basically erase it and its  paired element must be
                   // notified
      otherPairElement_->notifyDeath();
    otherPairElement_ = h.token_;
    h.token_->otherPairElement_ = this;
    h.invalidate(); // we need to leave the handle in an invalid state.
    return *this;
  }

  ~PairElementToken()
  {
    if(isPaired())
      otherPairElement_->notifyDeath();
  }

  /** Pair this with other. Throw if this object is already paired.*/
  void pairWith(PairElementToken & other)
  {
    if(isPaired() || other.isPaired())
      throw std::runtime_error("Can only pair if both tokens are unpaired.");
    otherPairElement_ = &other;
    other.otherPairElement_ = this;
  }

  /** Is this token paired? */
  bool isPaired() const { return otherPairElement_ != nullptr; }

  /** Is this token paired with other? */
  bool isPairedWith(const PairElementToken & other) const { return otherPairElement_ == &other; }

  /** Return the other element of the pair. Throw if this object is not paired*/
  const PairElementToken & otherPairElement() const
  {
    if(!isPaired())
      throw std::runtime_error("This token is not paired.");
    return *otherPairElement_;
  }

private:
  /** Create a token paired with \p other*/
  PairElementToken(PairElementToken * other) : otherPairElement_(other)
  {
    assert(!other->isPaired());
    other->otherPairElement_ = this;
  }

  void notifyMove(PairElementToken * newAddress) { otherPairElement_ = newAddress; }

  void notifyDeath() { otherPairElement_ = nullptr; }

  PairElementToken * otherPairElement_ = nullptr;
};

inline PairElementTokenHandle::PairElementTokenHandle(PairElementToken & t) noexcept : token_(&t)
{
  assert(!t.isPaired() && "Input token is already paired.");
}
} // namespace tvm::internal
