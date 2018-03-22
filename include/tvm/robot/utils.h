#pragma once

/* Copyright 2017-2018 CNRS-UM LIRMM, CNRS-AIST JRL
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

#include <tvm/api.h>
#include <tvm/Clock.h>
#include <tvm/Robot.h>

namespace tvm
{

namespace robot
{


/** Load a robot from a URDF file
 *
 * \p clock Clock tied to the robot
 *
 * \p name Name of the robot
 *
 * \p path Path to the URDF file
 *
 * \p fixed If true, load a fixed-based robot, otherwise add a free-flyer base
 *
 * \p fileteredLinks Ignore the links in this list when parsing the URDF file
 *
 * \p q Starting configuration of the robot
 *
 * \throws tvm::exception::DataException if \p path does not exist
 *
 */

std::unique_ptr<Robot> TVM_DLLAPI fromURDF(tvm::Clock & clock, const std::string & name,
                                           const std::string & path, bool fixed,
                                           const std::vector<std::string> & filteredLinks,
                                           const std::map<std::string, std::vector<double>> & q);

} // namespace robot

} // namespace tvm
