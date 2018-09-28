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

#include <tvm/api.h>
#include <tvm/defs.h>
#include <tvm/internal/ObjWithId.h>
#include <tvm/scheme/internal/SchemeAbilities.h>

namespace tvm
{

  namespace scheme
  {
    using identifier = int;

    namespace internal
    {
      /** Base class for solving a ControlProblem*/
      class TVM_DLLAPI ResolutionSchemeBase: public tvm::internal::ObjWithId
      {
      public:
        double big_number() const;
        void big_number(double big);

      protected:
        /** Constructor, meant only for derived classes
        *
        * \param abilities The set of abilities for this scheme.
        * \param big A big number use to represent infinity, in particular when
        * specifying non-existing bounds (e.g. x <= Inf is given as x <= big).
        */
        ResolutionSchemeBase(SchemeAbilities abilities, double big = constant::big_number);

        SchemeAbilities abilities_;

        /** A number to use for infinite bounds*/
        double big_number_;
      };
    }  // namespace internal

  }  // namespace scheme

}  // namespace tvm
