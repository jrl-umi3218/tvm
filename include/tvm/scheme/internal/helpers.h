#pragma once

/* Copyright 2017 CNRS-UM LIRMM, CNRS-AIST JRL
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with TVM.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace tvm
{
namespace scheme
{
namespace internal
{
  class ProblemComputationData;

  /** Get the computation data linked to a particular resolution scheme.
    * If this data does not exist, create it, using the resolution scheme as
    * a factory
    * \param The resolution scheme for which we want to get back the
    * computation data. It needs to have a method \a createComputationData.
    */
  template<typename Problem, typename Scheme>
  inline ProblemComputationData& getComputationData(Problem& problem, const Scheme& resolutionScheme);
}
}
}