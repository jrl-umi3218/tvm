#pragma once

#include <map>
#include <set>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace taskvm
{
  namespace internal
  {
    template <typename E>
    inline size_t enumId() 
    {
      static_assert(std::is_enum<E>::value, "this can only be called on enum and enum class.");
      return typeid(E).hash_code(); 
    }

    /** A class to convert any enum value in a unified type, losing the strong type 
      * but retaining the unicity of the value.
      * If we have two enums E1 = {u1=1, u2, ...} and E2 = {v1=1, v2, ...},
      * UnifiedEnumValue(u1) and UnifiedEnumValue(v1) have the same type but are
      * different.
      */
    class UnifiedEnumValue
    {
    public:
      template<typename E> UnifiedEnumValue(E e)
        : value_(enumId<E>(), static_cast<int>(e))
      {
        static_assert(std::is_enum<E>::value, "this can only be called on enum and enum class.");
      }

      /** Attempt to recast to enum E. Throw if this instance does not represent
        * a value of E
        */
      template <typename E>
      explicit operator E() const
      {
        static_assert(std::is_enum<E>::value, "this can only be called on enum and enum class.");
        if (enumId<E>() == value_.first)
          return static_cast<E>(value_.second);
        else
          throw std::logic_error("Cannot cast to this enum");
      }

      /** Check is the underlying enum type of this instance is E */
      template <typename E>
      bool hasEnumType()
      {
        return enumId<E>() == value_.second;
      }

      size_t enumTypeId()     const   { return value_.first; }
      int    bareEnumeValue() const   { return value_.second; }

    private:
      std::pair<size_t, int> value_;

      friend bool operator==(const UnifiedEnumValue&, const UnifiedEnumValue&);
      friend bool operator<(const UnifiedEnumValue&, const UnifiedEnumValue&);
    };



    //UNUSED for now
    /** While UnifiedEnumValue allows to mix different enums in a same container,
      * UnifiedEnumSet emulates a std::set<UnifiedValue> where all elements are
      * issued from the same enum. 
      * The enum type is specified by the first element inserted.
      */
    class UnifiedEnumSet
    {
    public:
      template <typename E>
      void insert(E e)
      {
        size_t id = enumId<E>();
        if (set_.empty())
        {
          enumId_ = id;
          set_.insert(static_cast<int>(e));
        }
        else
        {
          if (id == enumId_)
            set_.insert(static_cast<int>(e));
          else
            throw std::logic_error("You cannot insert an element of this enum in the set.");
        }
      }

      template <typename E>
      bool contains(E e) const
      {
        return enumId_ == enumId<E>() && set_.find(static_cast<int>(e)) != set_.end();
      }

    private:
      size_t        enumId_;
      std::set<int> set_;
    };



    inline bool operator==(const UnifiedEnumValue& v1, const UnifiedEnumValue& v2)
    {
      return v1.value_ == v2.value_;
    }

    inline bool operator<(const UnifiedEnumValue& v1, const UnifiedEnumValue& v2)
    {
      return v1.value_ < v2.value_;
    }
  }
}