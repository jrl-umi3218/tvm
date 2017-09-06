#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <Eigen/Core>

#include <tvm/data/Node.h>
#include "defs.h"
#include "MatrixProperties.h"

namespace tvm
{
  namespace internal
  {
    /* Describes an entity that can provide a value and its jacobian*/

    class TVM_DLLAPI FirstOrderProvider : public data::Node<FirstOrderProvider>
    {
    public:
      SET_OUTPUTS(FirstOrderProvider, Value, Jacobian)

      /** Note: by default, these methods return the cached value.
      * However, they are virtual in case the user might want to bypass the cache.
      * This would be typically the case if he/she wants to directly return the
      * output of another method, e.g. return the jacobian of an other Function.
      */
      virtual const Eigen::VectorXd& value() const;
      virtual const Eigen::MatrixXd& jacobian(const Variable& x) const;

      /** Return the output size m*/
      int size() const;

      /** Return the variables*/
      const std::vector<VariablePtr>& variables() const;

      /** Properties of the jacobian matrix related to variable x.*/
      const MatrixProperties& jacobianProperties(const Variable& x) const;

    protected:
      FirstOrderProvider(int m);

      /** Resize all cache members corresponding to active outputs.
        *
        * This can be overriden in case you do not need all of the default
        * mechanism (typically if you will not use part of the cache).
        * If you override to perform additional operations, do not forget to 
        * call this base version in the derived classes.
        */
      virtual void resizeCache();

      /** Sub-methods of resizeCache to be used by derived classes that need 
        * this level of granularity.
        */
      void resizeValueCache();
      void resizeJacobianCache();

      /** Add or remove variables. Cache is automatically updated*/
      void addVariable(VariablePtr);
      void removeVariable(VariablePtr);

      /** To be overriden by derived classes that need to react to
        * the addition of a variable. Called at the end of addVariable();
        */
      virtual void addVariable_(VariablePtr);
      virtual void removeVariable_(VariablePtr);

      /** Setter for the properties if the jacobian matrix related to variable x.*/
      void jacobianProperties(const Variable& x, MatrixProperties p);

      // cache
      Eigen::VectorXd value_;
      std::map<Variable const*, Eigen::MatrixXd> jacobian_;

    private:
      int m_; //output size
      std::vector<VariablePtr> variables_;
      std::map<Variable const*, MatrixProperties> jacobianProperties_;
    };


    inline const Eigen::VectorXd& FirstOrderProvider::value() const
    {
      return value_;
    }

    inline const Eigen::MatrixXd& FirstOrderProvider::jacobian(const Variable& x) const
    {
      return jacobian_.at(&x);
    }

    inline int FirstOrderProvider::size() const
    {
      return m_;
    }

    inline const std::vector<VariablePtr>& FirstOrderProvider::variables() const
    {
      return variables_;
    }

    inline const MatrixProperties& FirstOrderProvider::jacobianProperties(const Variable& x) const
    {
      return jacobianProperties_.at(&x);
    }
  }
}