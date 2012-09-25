#pragma once
#ifndef OPENGM_TRIBOOL_HXX
#define OPENGM_TRIBOOL_HXX

namespace opengm {

   /// Variable with three values (true=1, false=0, maybe=-1)
   class Tribool
   {
   public:
      enum State {True=1, False=0, Maybe=-1};

      Tribool();
      Tribool(const Tribool&);
      template<class T>
         Tribool(const T);
      Tribool(Tribool::State state);

      Tribool& operator=(const Tribool&);
      template<class T>
         Tribool& operator=(T);
      Tribool& operator=(Tribool::State state);

      bool operator==(const bool a) const;
      template<class T>
         bool operator==(T a) const;
      bool operator!=(const bool a) const;
      operator bool() const;
      bool operator!() const;
      bool maybe() const;
      
      void operator&=(Tribool::State state);

   private:
      char value_;
      friend std::ostream& operator<<(std::ostream& out, const Tribool& t );
   };

   inline Tribool::Tribool()
   :  value_(Tribool::Maybe)
   {}

   inline Tribool::Tribool
   (
      const Tribool& val
   )
   :  value_(val.value_)
   {}

   inline Tribool::Tribool
   (
      Tribool::State state
   )
   :  value_(state)
   {}

   template<class T>
   inline Tribool::Tribool
   (
      const T val
   )
   :  value_(static_cast<char>(val) == Tribool::Maybe 
             ? Tribool::Maybe 
             : static_cast<char>(static_cast<bool>(val)))
   {}

   inline Tribool& 
   Tribool::operator=
   (
      const Tribool& rhs
   )
   {
      if(this != &rhs) {
         value_ = rhs.value_;
      }
      return *this;
   }

   template<class T>
   inline Tribool& 
   Tribool::operator=
   (
      const T val
   )
   {
      static_cast<char>(val) == Tribool::Maybe 
         ? value_ = Tribool::Maybe 
         : value_ = static_cast<char>(static_cast<bool>(val));
      return *this;
   }

   inline Tribool& 
   Tribool::operator=
   (
      Tribool::State val
   )
   {
      value_ = static_cast<char>(val);
      return *this;
   }

   inline bool 
   Tribool::operator==
   (
      const bool a
   ) const
   {
      return bool( (value_ == Tribool::True && a == true)
         || (value_ == Tribool::False && a == false));
   }

   template<class T>
   inline bool 
   Tribool::operator==
   (
      T a
   ) const
   {
      return static_cast<char>(a) == value_;
   }

   inline bool 
   Tribool::operator!=
   (
      const bool a
   ) const
   {
      return (value_ != Tribool::True && a == true)
         || (value_ != Tribool::True && a == false);
   }

   inline Tribool::operator bool() const
   {
      return value_ == Tribool::True;
   }

   inline bool 
   Tribool::operator!() const
   {
      return value_ == Tribool::False;
   }

   inline bool 
   Tribool::maybe() const
   {
      return value_ == Tribool::Maybe;
   }

   inline std::ostream& 
   operator<<
   (
      std::ostream& out, 
      const Tribool& t
   )
   {
      out << static_cast<int>(t.value_);
      return out;
   } 

   inline void 
   Tribool::operator&=(Tribool::State state)
   {
      if(state==Tribool::True && value_!=Tribool::False) value_=Tribool::True;
      if(state==Tribool::False) value_=Tribool::False;                
   }

} // namespace opengm

#endif // #ifndef OPENGM_TRIBOOL_HXX

