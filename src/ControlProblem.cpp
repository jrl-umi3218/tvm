/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its contributors
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include <tvm/ControlProblem.h>

namespace tvm
{

  TaskWithRequirements::TaskWithRequirements(const Task& t, requirements::SolvingRequirements req)
    : task(t)
    , requirements(req)
  {
  }


  TaskWithRequirementsPtr ControlProblem::add(const Task& task, const requirements::SolvingRequirements& req)
  {
    auto tr = std::make_shared<TaskWithRequirements>(task, req);
    add(tr);
    return tr;
  }

  void ControlProblem::add(TaskWithRequirementsPtr tr)
  {
    tr_.push_back(tr);
    addCallBackToTask(tr);
    notify(scheme::internal::ProblemDefinitionEvent(scheme::internal::ProblemDefinitionEvent::Type::TaskAddition, tr.get()));
    finalized_ = false;
  }

  void ControlProblem::remove(TaskWithRequirements* tr)
  {
    auto it = std::find_if(tr_.begin(), tr_.end(), [tr](const TaskWithRequirementsPtr& p) {return p.get() == tr; });
    if (it != tr_.end())
      tr_.erase(it);
    notify(scheme::internal::ProblemDefinitionEvent(scheme::internal::ProblemDefinitionEvent::Type::TaskRemoval, tr));
    callbackTokens_.erase(tr);
    finalized_ = false;
  }

  const std::vector<TaskWithRequirementsPtr>& ControlProblem::tasks() const
  {
    return tr_;
  }

  int ControlProblem::size() const
  {
    return static_cast<int>(tr_.size());
  }

  void ControlProblem::update()
  {
    finalize();
    updater_.run();
    update_();
  }

  void ControlProblem::finalize()
  {
    if (!finalized_)
    {
      updater_.refresh();
      finalize_();
      finalized_ = true;
    }
  }

  void ControlProblem::notify(const scheme::internal::ProblemDefinitionEvent& e)
  {
    for (auto& c : computationData_)
    {
      c.second->addEvent(e);
    }
  }

  void ControlProblem::addCallBackToTask(TaskWithRequirementsPtr tr)
  {
    using EventType = scheme::internal::ProblemDefinitionEvent::Type;
    std::vector<internal::PairElementToken> tokens;
    TaskWithRequirements* t = tr.get();
    
    auto l1 = [this, t]() {this->notify({ EventType::WeightChange, t }); };
    tokens.emplace_back(tr->requirements.weight().registerCallback(l1));
    
    auto l2 = [this, t]() {this->notify({ EventType::AnisotropicWeightChange, t }); };
    tokens.emplace_back(tr->requirements.anisotropicWeight().registerCallback(l2));

    callbackTokens_[t] = std::move(tokens);
  }


  ControlProblem::Updater::Updater()
    : upToDate_(false)
    , inputs_(new graph::internal::Inputs)
  {
  }

  void ControlProblem::Updater::refresh()
  {
    if (!upToDate_)
    {
      updateGraph_.clear();
      updateGraph_.add(inputs_);
      updateGraph_.update();
      upToDate_ = true;
    }
  }

  void ControlProblem::Updater::run()
  {
    updateGraph_.execute();
  }

}  // namespace tvm
