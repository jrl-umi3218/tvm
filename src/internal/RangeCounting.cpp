#include <tvm/internal/RangeCounting.h>

namespace tvm::internal
{
void RangeCounting::add(const Range & r)
{
  if(r.dim == 0)
    return;

  recompute_ = true;
  if(limits_.empty())
  {
    limits_.emplace_back(r.start, true);
    limits_.emplace_back(r.start + r.dim, false);
  }
  else
  {
    Limit l = {r.start, true};
    Limit u = {r.start + r.dim, false};
    auto it = limits_.cbegin();
    for(; it != limits_.cend() && it->i_ <= l.i_; ++it) {}
    if(it == limits_.cend())
    {
      limits_.push_back(l);
      limits_.push_back(u);
      return;
    }
    if(it->i_ == l.i_)
    {
      assert(!it->lower_);
      it = limits_.erase(it);
    }
    else
      limits_.insert(it, l);
    for(; it != limits_.cend() && (*it) < u; ++it) {}
    if(auto itp = std::prev(it); itp->i_ == u.i_)
    {
      assert(itp->lower_);
      limits_.erase(itp);
    }
    else
      limits_.insert(it, u);
  }
}

void RangeCounting::remove(const Range & r)
{
  if(r.dim == 0)
    return;

  recompute_ = true;
  int stop = r.start + r.dim;
  if(limits_.empty())
    throw std::runtime_error("Cannot remove interval on empty list");
  if(r.start < limits_.front().i_ || r.start >= limits_.back().i_ || stop <= limits_.front().i_ || stop > limits_.back().i_)
    throw std::runtime_error("The removed interval must be included in the current intervals");

  int depth = 0;
  Limit l(r.start, false);
  auto it = limits_.cbegin();
  for(; it != limits_.cend(); ++it)
  {
    if(l < *it)
      break;
    if(it->lower_)
      ++depth;
    else
      --depth;
  }
  auto itp = std::prev(it);
  if(itp->i_ == l.i_)
  {
    assert(itp->lower_);
    limits_.erase(itp);
    --depth;
  }
  else
  {
    if(depth == 0)
      throw std::runtime_error("Trying to remove from a non-existing interval");
    limits_.insert(it, l);
    --depth;
  }

  Limit u(stop, true);
  for(; it != limits_.cend(); ++it)
  {
    if(u < *it)
      break;
    if(it->lower_)
      ++depth;
    else
      --depth;
    if(depth < 0)
      throw std::runtime_error("Trying to remove from a non-existing interval");
  }
  if(it->i_ == u.i_)
  {
    assert(!it->lower_);
    limits_.erase(it);
    ++depth;
  }
  else
  {
    limits_.insert(it, l);
  }
}

const std::vector<Range> & RangeCounting::ranges() const
{
  if(recompute_)
  {
    intervals_.clear();
    int depth = 0;
    auto s = limits_.cbegin();
    for(auto it = limits_.cbegin(); it != limits_.cend(); ++it)
    {
      if(it->lower_)
        ++depth;
      else
        --depth;
      if(depth == 0)
      {
        intervals_.emplace_back(s->i_, it->i_ - s->i_);
        s = std::next(it);
      }
    }
    recompute_ = false;
  }

  return intervals_;
}

const std::list<RangeCounting::Limit> & RangeCounting::limits() const { return limits_; }

} // namespace tvm::internal
