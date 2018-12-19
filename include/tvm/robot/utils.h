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

std::unique_ptr<Robot> TVM_DLLAPI fromURDF(tvm::Clock & clock, const std::string & name,
                                           const std::string & path, bool fixed,
                                           const std::vector<std::string> & filteredLinks,
                                           const std::map<std::string, std::vector<double>> & q);

} // namespace robot

} // namespace tvm
