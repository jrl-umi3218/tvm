/** Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM */

#pragma once

#include <tvm/api.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace tvm
{

namespace graph
{
class CallGraph;

namespace internal
{
class Inputs;

/** lexicographic comparison of two objects given an ordered list of the
 * members to compare.
 */
template<typename ObjType, typename MemberType, typename... Args>
bool lexLess(const ObjType & l, const ObjType & r, MemberType ObjType::* member, Args &&... args)
{
  return (l.*member) < (r.*member) || ((l.*member == r.*member) && lexLess(l, r, std::forward<Args>(args)...));
}

/** Equality comparison of two objects given an ordered list of the members
 * to compare.
 */
template<typename ObjType, typename MemberType, typename... Args>
bool eq(const ObjType & l, const ObjType & r, MemberType ObjType::* member, Args &&... args)
{
  return (l.*member == r.*member) && eq(l, r, std::forward<Args>(args)...);
}

/** End of recursion. */
template<typename ObjType, typename MemberType>
bool lexLess(const ObjType & l, const ObjType & r, MemberType ObjType::* member)
{
  return (l.*member) < (r.*member);
}

/** End of recursion. */
template<typename ObjType, typename MemberType>
bool eq(const ObjType & l, const ObjType & r, MemberType ObjType::* member)
{
  return (l.*member) == (r.*member);
}

/** A data structure, used by Logger to log the inputs, outputs, updates and
 * dependencies that are declared at runtime.
 */
class TVM_DLLAPI Log
{
public:
  /** A non-enum representation of enum, as a pair (type of enum, value).*/
  struct EnumValue
  {
    /** Build from an enum*/
    template<typename E>
    EnumValue(E e);

    std::type_index type; // representation of the enumeration type
    int value;            // value of the enumeration

    bool operator<(const EnumValue & other) const { return lexLess(*this, other, &EnumValue::type, &EnumValue::value); }
    bool operator==(const EnumValue & other) const { return eq(*this, other, &EnumValue::type, &EnumValue::value); }
  };

  /** A type-independent representation of a pointer as a pair (type, address).*/
  struct Pointer
  {
    /** Build from an pointer*/
    template<typename T>
    Pointer(T * p);

    /** Build from a pair (type, address).*/
    Pointer(const std::type_index & t, std::uintptr_t v);

    std::type_index type; // representation of the pointer type
    std::uintptr_t value; // address of the pointer

    bool operator<(const Pointer & other) const { return lexLess(*this, other, &Pointer::value); }
    bool operator==(const Pointer & other) const { return eq(*this, other, &Pointer::value); }
  };

  /** Description of an update. */
  struct Update
  {
    EnumValue id;            // id of the update
    std::string name;        // name of the update
    std::uintptr_t function; // address of the update function
    Pointer owner;           // address of the instance registering the update
    bool operator<(const Update & other) const
    {
      return lexLess(*this, other, &Update::owner, &Update::id, &Update::function);
    }
    bool operator==(const Update & other) const
    {
      return eq(*this, other, &Update::owner, &Update::id, &Update::function);
    }
  };

  /** Description of an output. */
  struct Output
  {
    EnumValue id;     // id of the output
    std::string name; // name of the output
    Pointer owner;    // address of the instance registering the output
    bool operator<(const Output & other) const { return lexLess(*this, other, &Output::owner, &Output::id); }
    bool operator==(const Output & other) const { return eq(*this, other, &Output::owner, &Output::id); }
  };

  /** Description of an input. */
  struct Input
  {
    EnumValue id;     // id of the input
    std::string name; // name of the input
    Pointer source;   // address of the instance providing the input
    Pointer owner;    // address of the instance registering the input
    bool operator<(const Input & other) const
    {
      return lexLess(*this, other, &Input::owner, &Input::id, &Input::source);
    }
    bool operator==(const Input & other) const { return eq(*this, other, &Input::owner, &Input::id, &Input::source); }
  };

  /** Description of an input->update dependency. */
  struct InputDependency
  {
    EnumValue input;  // the input
    EnumValue update; // the update
    Pointer source;   // address of the instance providing the input
    Pointer owner;    // address of the instance registering the dependency
    bool operator<(const InputDependency & other) const
    {
      return lexLess(*this, other, &InputDependency::owner, &InputDependency::update, &InputDependency::source,
                     &InputDependency::input);
    }
    bool operator==(const InputDependency & other) const
    {
      return eq(*this, other, &InputDependency::owner, &InputDependency::update, &InputDependency::source,
                &InputDependency::input);
    }
  };

  /** Description of an update->output dependency. */
  struct OutputDependency
  {
    EnumValue update; // the update
    EnumValue output; // the output
    Pointer owner;    // address of the instance registering the dependency
    bool operator<(const OutputDependency & other) const
    {
      return lexLess(*this, other, &OutputDependency::owner, &OutputDependency::update, &OutputDependency::output);
    }
    bool operator==(const OutputDependency & other) const
    {
      return eq(*this, other, &OutputDependency::owner, &OutputDependency::update, &OutputDependency::output);
    }
  };

  /** Description of an update->update dependency. */
  struct InternalDependency
  {
    EnumValue from; // the update depended upon
    EnumValue to;   // the depending update
    Pointer owner;  // address of the instance registering the dependency
    bool operator<(const InternalDependency & other) const
    {
      return lexLess(*this, other, &InternalDependency::owner, &InternalDependency::from, &InternalDependency::to);
    }
    bool operator==(const InternalDependency & other) const
    {
      return eq(*this, other, &InternalDependency::owner, &InternalDependency::from, &InternalDependency::to);
    }
  };

  /** Description of an input->output dependency. */
  struct DirectDependency
  {
    EnumValue input;  // the input
    EnumValue output; // the output
    Pointer source;   // address of the instance providing the input
    Pointer owner;    // address of the instance registering the dependency
    bool operator<(const DirectDependency & other) const
    {
      return lexLess(*this, other, &DirectDependency::owner, &DirectDependency::output, &DirectDependency::source,
                     &DirectDependency::input);
    }
    bool operator==(const DirectDependency & other) const
    {
      return eq(*this, other, &DirectDependency::owner, &DirectDependency::output, &DirectDependency::source,
                &DirectDependency::input);
    }
  };

  /** Build the list of updates and outputs used in the CallGraph \p g*/
  std::pair<std::vector<Log::Output>, std::vector<Log::Update>> subGraph(const CallGraph * const g) const;

  /** Build the list of updates and outputs used from \p out*/
  std::pair<std::vector<Log::Output>, std::vector<Log::Update>> subGraph(const Output out) const;

  /** Generate a dot representation for node corresponding to p.*/
  std::string generateDot(const Pointer & p) const;

  /** Generate the specified CallGraph. */
  std::string generateDot(const CallGraph * const g) const;

  /** Generate the whole graph, highlighting the elements specified by
   * oUtHighlight and upHightlight.
   */
  std::string generateDot(const std::vector<Log::Output> & outHighlight = {},
                          const std::vector<Log::Update> & upHighlight = {}) const;

  /** Get the most derived type corresponding to pointer \p p.
   *
   * (This is based on the assumption that the log add the most derived type last)
   */
  const std::type_index & getPromotedType(const Pointer & p) const;

  // raw logs
  std::vector<Update> updates_;
  std::vector<Input> inputs_;
  std::vector<Output> outputs_;
  std::vector<InputDependency> inputDependencies_;
  std::vector<OutputDependency> outputDependencies_;
  std::vector<InternalDependency> internalDependencies_;
  std::vector<DirectDependency> directDependencies_;

  /** Each elements of the map is the list of inputs of a CallGraph, as added
   * in CallGraph.add
   */
  std::map<Pointer, std::vector<Pointer>> graphOutputs_;

  /** Maps a data address to all the type info associated with this address.
   * For a given address, each type appears only once, and types are sorted in
   * their order of appearance in the logging process. We assume that the
   * last one to appear is the most derived in the inheritance hierarchy.
   *
   * FIXME: can we find a more robust way to determine the real type of a
   * data?
   */
  std::map<std::uintptr_t, std::vector<std::type_index>> types_;

private:
  std::pair<std::vector<Log::Output>, std::vector<Log::Update>> followUpDependency(
      const std::vector<Output> & allOutputs,
      const std::vector<Output> & startingPoints) const;

  /** Generate a (unique) name for the given output, based on its name, and
   * the class and memory address of its owner.
   */
  std::string nodeName(const Log::Output & output) const;
  /** Generate a (unique) name for the given input, based on its name, and
   * the class and memory address of the source.
   */
  std::string nodeName(const Log::Input & input) const;
  /** Generate a (unique) name for the given update, based on its name, and
   * the class and memory address of its owner.
   */
  std::string nodeName(const Log::Update & update) const;
};

template<typename E>
inline Log::EnumValue::EnumValue(E e) : type(typeid(e)), value(static_cast<int>(e))
{
  static_assert(std::is_enum<E>::value, "EnumValue can only be defined for enums.");
}

template<typename T>
inline Log::Pointer::Pointer(T * p) : type(typeid(*p)), value(reinterpret_cast<std::uintptr_t>(p))
{}

inline Log::Pointer::Pointer(const std::type_index & t, std::uintptr_t v) : type(t), value(v) {}

} // namespace internal

} // namespace graph

} // namespace tvm
