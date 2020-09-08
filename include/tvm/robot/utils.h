/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

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

std::unique_ptr<Robot> TVM_DLLAPI fromURDF(tvm::Clock & clock,
                                           const std::string & name,
                                           const std::string & path,
                                           bool fixed,
                                           const std::vector<std::string> & filteredLinks,
                                           const std::map<std::string, std::vector<double>> & q);

} // namespace robot

} // namespace tvm
