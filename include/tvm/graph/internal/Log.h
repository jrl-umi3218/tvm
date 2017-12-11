#pragma once

#include <tvm/api.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#define LOG_GRAPH

namespace tvm
{
  
namespace graph
{

namespace internal
{
  class Inputs;

  //lexicographic comparison of two objects given an ordered list of the members to compare
  template<typename ObjType, typename MemberType, typename... Args>
  bool lexLess(const ObjType& l, const ObjType& r, MemberType ObjType::* member, Args&&... args)
  {
    return (l.*member) < (r.*member) || ((l.*member == r.*member) && lexLess(l, r, std::forward<Args>(args)...));
  }

  //equality comparison of two objects given an ordered list of the members to compare
  template<typename ObjType, typename MemberType, typename... Args>
  bool eq(const ObjType& l, const ObjType& r, MemberType ObjType::* member, Args&&... args)
  {
    return (l.*member == r.*member) && eq(l, r, std::forward<Args>(args)...);
  }

  //end of recursion
  template<typename ObjType, typename MemberType>
  bool lexLess(const ObjType& l, const ObjType& r, MemberType ObjType::* member)
  {
    return (l.*member) < (r.*member);
  }

  //end of recursion
  template<typename ObjType, typename MemberType>
  bool eq(const ObjType& l, const ObjType& r, MemberType ObjType::* member)
  {
    return (l.*member) == (r.*member);
  }

  class TVM_DLLAPI Log
  {
  public:
    struct TVM_DLLAPI TypeInfo
    {
      TypeInfo(const std::type_info& t);

      size_t hash;
      std::string name;

      bool operator<(const TypeInfo& other) const { return hash < other.hash; }
      bool operator==(const TypeInfo& other) const { return hash == other.hash; }
    };

    struct EnumValue
    {
      template<typename E>
      EnumValue(E e);

      TypeInfo type;
      int value;
      
      bool operator<(const EnumValue& other) const { return lexLess(*this, other, &EnumValue::type, &EnumValue::value); }
      bool operator==(const EnumValue& other) const { return eq(*this, other, &EnumValue::type, &EnumValue::value); }
    };

    struct Pointer
    {
      template<typename T> Pointer(T* p);

      Pointer(const TypeInfo& t, std::uintptr_t v);

      TypeInfo type;
      std::uintptr_t value;

      bool operator<(const Pointer& other) const { return lexLess(*this, other, &Pointer::type, &Pointer::value); }
      bool operator==(const Pointer& other) const { return eq(*this, other, &Pointer::type, &Pointer::value); }
    };

    struct Update
    {
      EnumValue id;             //id of the update
      std::string name;         //name of the update
      std::type_index typeT;    //type of the node
      std::type_index typeU;    //type of the class registering the update
      std::uintptr_t function;  //address of the update function
      Pointer owner;            //address of the instance registering the update
      bool operator<(const Update& other) const { return lexLess(*this, other, &Update::owner, &Update::id, &Update::function); }
      bool operator==(const Update& other) const { return eq(*this, other, &Update::owner, &Update::id, &Update::function); }
    };

    struct Output
    {
      EnumValue id;             //id of the output
      std::string name;         //name of the output
      Pointer owner;            //address of the instance registering the output
      bool operator<(const Output& other) const { return lexLess(*this, other, &Output::owner, &Output::id); }
      bool operator==(const Output& other) const { return eq(*this, other, &Output::owner, &Output::id); }
    };

    struct Input
    {
      EnumValue id;             //id of the input
      std::string name;         //name of the input
      Pointer source;           //address of the instance providing the input
      Pointer owner;            //address of the instance registering the input
      bool operator<(const Input& other) const { return lexLess(*this, other, &Input::owner, &Input::id, &Input::source); }
      bool operator==(const Input& other) const { return eq(*this, other, &Input::owner, &Input::id, &Input::source); }
    };

    struct InputDependency
    {
      EnumValue input;
      EnumValue update;
      Pointer source;           //address of the instance providing the input
      Pointer owner;            //address of the instance registering the dependency
      bool operator<(const InputDependency& other) const { return lexLess(*this, other, &InputDependency::owner, &InputDependency::update, &InputDependency::source, &InputDependency::input); }
      bool operator==(const InputDependency& other) const { return eq(*this, other, &InputDependency::owner, &InputDependency::update, &InputDependency::source, &InputDependency::input); }
    };

    struct OutputDependency
    {
      EnumValue update;
      EnumValue output;
      Pointer owner;            //address of the instance registering the dependency
      bool operator<(const OutputDependency& other) const { return lexLess(*this, other, &OutputDependency::owner, &OutputDependency::update, &OutputDependency::output); }
      bool operator==(const OutputDependency& other) const { return eq(*this, other, &OutputDependency::owner, &OutputDependency::update, &OutputDependency::output); }
    };

    struct InternalDependency
    {
      EnumValue from;
      EnumValue to;
      Pointer owner;            //address of the instance registering the dependency
      bool operator<(const InternalDependency& other) const { return lexLess(*this, other, &InternalDependency::owner, &InternalDependency::from, &InternalDependency::to); }
      bool operator==(const InternalDependency& other) const { return eq(*this, other, &InternalDependency::owner, &InternalDependency::from, &InternalDependency::to); }
    };

    struct DirectDependency
    {
      EnumValue input;
      EnumValue output;
      Pointer source;           //address of the instance providing the input
      Pointer owner;            //address of the instance registering the dependency
      bool operator<(const DirectDependency& other) const { return lexLess(*this, other, &DirectDependency::owner, &DirectDependency::output, &DirectDependency::source, &DirectDependency::input); }
      bool operator==(const DirectDependency& other) const { return eq(*this, other, &DirectDependency::owner, &DirectDependency::output, &DirectDependency::source, &DirectDependency::input); }
    };

    /** Generate a dot representation for node corresponding to p.*/
    std::string generateDot(const Pointer& p) const;

    /** Generate the whole graph*/
    std::string generateDot() const;

    //raw logs
    std::vector<Update> updates_;
    std::vector<Input> inputs_;
    std::vector<Output> outputs_;
    std::vector<InputDependency> inputDependencies_;
    std::vector<OutputDependency> outputDependencies_;
    std::vector<InternalDependency> internalDependencies_;
    std::vector<DirectDependency> directDependencies_;

    /** Maps data adress to all the type info associated with this adress.
      * For a given adress, each type appears only once, and types are sorted in
      * their order of appearance in the logging process. We assume that the
      * last one to appear is the most derived in the inheritance hierarchy.
      */
    std::map<std::uintptr_t, std::vector<TypeInfo>> types_;

    private:
      std::string nodeName(const Log::Output& output) const;
      std::string nodeName(const Log::Input& input) const;
      std::string nodeName(const Log::Update& update) const;
  };


  template<typename E>
  inline Log::EnumValue::EnumValue(E e)
    : type(typeid(e)), value(static_cast<int>(e))
  {
    //static_assert(std::is_enum<E>::value);
  }

  template<typename T>
  inline Log::Pointer::Pointer(T* p)
    : type(typeid(*p))
    , value(reinterpret_cast<std::uintptr_t>(p))
  {
  }

  inline Log::Pointer::Pointer(const TypeInfo & t, std::uintptr_t v)
    : type(t), value(v)
  {
  }

} // namespace internal

}  //namespace graph

} // namespace tvm
