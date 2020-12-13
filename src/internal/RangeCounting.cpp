#include <tvm/internal/RangeCounting.h>

namespace tvm::internal
{
bool RangeCounting::add(const Range & r)
{
  assert(isValid());
  if(r.dim == 0)
    return false;

  recomputeSplit_ = true;
  if(limits_.empty())
  {
    limits_.emplace_back(r.start, Limit::Lower);
    limits_.emplace_back(r.end(), Limit::Upper);
    return recompute(true);
  }
  else
  {
    int depth = 0;
    bool change = false;
    Limit l = {r.start, Limit::Lower};
    Limit u = {r.end(), Limit::Upper};
    auto it = limits_.begin();
    moveToFirstAfter(l, it, depth);
    if(it == limits_.end())
    {
      limits_.push_back(l);
      limits_.push_back(u);
      return recompute(true);
    }
    insert(l, it, depth);
    if(depth == 1)
      change = true;

    change = moveToFirstAfter(Limit(r.end(), Limit::Cut), it, depth, 1) || change;
    insert(u, it, depth);
    return recompute(change);
  }
}

bool RangeCounting::remove(const Range & r)
{
  assert(isValid());
  if(r.dim == 0)
    return false;

  if(limits_.empty())
    throw std::runtime_error("Cannot remove interval on empty list");
  if(r.start < limits_.front().i_ || r.start >= limits_.back().i_ || r.end() <= limits_.front().i_
     || r.end() > limits_.back().i_)
    throw std::runtime_error("The removed interval must be included in the current intervals");

  recomputeSplit_ = true;
  int depth = 0;
  Limit l(r.start, Limit::Upper);
  Limit u(r.end(), Limit::Lower);
  auto it = limits_.begin();
  moveToFirstAfter({r.start, Limit::Cut}, it, depth);
  if(depth == 0)
    throw std::runtime_error("Trying to remove from a non-existing interval");
  bool change = (depth == 1);
  auto it1 = it; // We don't insert the value yet, to keep a valid state in case a exception is thrown
  int depth1 = depth;
  change = moveToFirstAfter(u, it, depth, 1) || change;
  insert(l, it1, depth1);
  --depth;
  insert(u, it, depth);
  depth = 0;
  it = limits_.begin();
  while(it != limits_.end()) // ideally from it1 to it, but the second insert can invalidate it1
  {
    depth -= it->type_;
    if(depth == 0 && it->type_ == Limit::Cut)
      it = limits_.erase(it);
    else
      ++it;
  }

  return recompute(change);
}

const std::vector<Range> & RangeCounting::ranges(bool splitOncountDiff) const
{
  if(recompute_ || recomputeSplit_)
  {
    intervals_.clear();
    int depth = 0;
    if(limits_.empty())
    {
      recompute_ = false;
      recomputeSplit_ = false;
      return intervals_;
    }
    if(splitOncountDiff)
    {
      auto next = [&depth, this](auto & it) {
        const Limit & v = *it;
        while(it != limits_.end() && *it == v)
        {
          depth -= it->type_;
          ++it;
        }
      };
      auto s = limits_.cbegin();
      auto it = s;
      next(it);
      while(it != limits_.end())
      {
        if(s->i_ != it->i_)
          intervals_.emplace_back(s->i_, it->i_ - s->i_);
        else
          assert((s->type_ == Limit::Lower && it->type_ == Limit::Cut)
                 || (s->type_ == Limit::Cut && it->type_ == Limit::Upper));
        s = it;
        next(it);
        if(depth == 0 && it != limits_.end())
        {
          s = it;
          next(it);
        }
      }
      recompute_ = true;
      recomputeSplit_ = false;
    }
    else
    {
      auto s = limits_.cbegin();
      for(auto it = limits_.cbegin(); it != limits_.cend(); ++it)
      {
        depth -= it->type_;
        if(depth == 0)
        {
          intervals_.emplace_back(s->i_, it->i_ - s->i_);
          s = std::next(it);
        }
      }
      recompute_ = false;
      recomputeSplit_ = true;
    }
  }

  return intervals_;
}

const std::list<RangeCounting::Limit> & RangeCounting::limits() const { return limits_; }

bool RangeCounting::moveToFirstAfter(const Limit & val, It & it, int & depth, int depthCut) const
{
  bool hit = false;
  while(it != limits_.end() && *it <= val)
  {
    depth -= it->type_;
    if(depth < depthCut)
      throw std::runtime_error("Too many upper limits. Are you trying to remove values that are not present?");
    if(depth == depthCut)
      hit = true;
    ++it;
  }
  return hit;
}

void RangeCounting::insert(const Limit & val, It & it, int & depth)
{
  if(val.type_ == Limit::Lower)
  {
    if(val.i_ == it->i_)
    {
      assert(it->type_ != Limit::Lower);
      if(it->type_ == Limit::Cut)
      {
        if(auto n = std::next(it); n != limits_.end() && *n == Limit(val.i_, Limit::Upper))
        {
          // insertion of (i,+) before (i,|) (i,-): (i,+) is not inserted and (i,-) is removed.
          if(depth == 0)
            limits_.erase(it);
          it = limits_.erase(n);
        }
        else
        {
          // insertion of (i,+) before (i,|)
          limits_.insert(it, val);
          ++depth;
        }
      }
      else
      {
        // insertion of (i,+) before (i,-): (i,-) is changed to (i,|)
        if(depth == 0)
          it = limits_.erase(it);
        else
        {
          it->type_ = Limit::Cut;
          ++it;
        }
      }
    }
    else
    {
      // insertion of (i,+)
      limits_.insert(it, val);
      ++depth;
    }
  }
  else
  {
    assert(val.type_ == Limit::Upper);
    assert(it != limits_.begin());
    auto p = std::prev(it);
    assert(!(*p == val));
    if(val.i_ == p->i_)
    {
      if(p->type_ == Limit::Cut)
      {
        if(p != limits_.begin() && *std::prev(p) == Limit(val.i_, Limit::Lower))
        {
          // insertion of (i,-) after (i,+) (i,|) : (i,-) is not inserted and (i,+) is removed
          limits_.erase(std::prev(p));
          if(depth == 0)
            it = limits_.erase(p);
          --depth;
        }
        else
        {
          // insertion of (i,-) after (i,|)
          limits_.insert(it, val);
          --depth;
        }
      }
      else
      {
        // insertion of (i,-) after (i,+): (i,+) is changed to (i,|)
        if(depth == 1)
          limits_.erase(p);
        else
          p->type_ = Limit::Cut;
        --depth;
      }
    }
    else
    {
      // insertion of (i,-)
      limits_.insert(it, val);
      --depth;
    }
  }
}

bool RangeCounting::recompute(bool change)
{
  recompute_ = recompute_ || change;
  return change;
}

bool RangeCounting::isValid() const
{
  if(limits_.size() == 0)
    return true;
  else if(limits_.size() == 1)
    return false;
  if(limits_.size() == 2)
    return *limits_.begin() < *std::next(limits_.begin());

  auto it1 = limits_.begin(), it2 = std::next(it1);
  if(*it1 > *it2 || (*it1 == *it2 && it1->type_ == Limit::Cut))
    return false;
  for(auto it3 = std::next(it2); it3 != limits_.end(); ++it1, ++it2, ++it3)
  {
    if(*it2 > *it3 || (*it2 == *it3 && it2->type_ == Limit::Cut))
      return false;
    if(it1->i_ == it2->i_ && it1->i_ == it3->i_ && it1->type_ == Limit::Lower && it2->type_ == Limit::Cut
       && it3->type_ == Limit::Upper)
      return false;
  }

  int depth = 0;
  for(auto it = limits_.begin(); it != limits_.end(); ++it)
  {
    if(it->type_ == Limit::Cut && depth == 0)
      return false;
    depth -= it->type_;
  }
  return true;
}

} // namespace tvm::internal
