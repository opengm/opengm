#ifndef RESTICTOR_HXX
#define	RESTICTOR_HXX

class RestrictionBase{
   virtual bool   isRestricted()    const=0;
   virtual bool   canBeRestricted() const=0;
   std::string    name()            const=0;
   std::string    description()     const=0;
};


class Restrictor{
public:
   Restrictor();
   ~Restrictor();
   // add restriction
   void     addRestriction(RestrictionBase *);
   
   // query
   bool        isRestricted()                const;
   bool        canBeResticed()               const;
   size_t      numberOfRestrictions()        const;
   bool        isRestricted(const size_t)    const;
   bool        canBeRestricted(const size_t) const;
   std::string name(const size_t )           const;
   std::string description(const size_t )    const;
private:
   std::vector<RestrictionBase *> restrictons_;
};

inline
Restrictor::Restrictor() {

}

Restrictor::~Restrictor() {
   for(size_t i=0;i<restrictons_.size();++i)
      delete restrictons_[i];
}
// add restriction
inline void 
Restrictor::addRestriction(RestrictionBase * restriction){
   restrictons_.push_back(restriction);
}

// query
inline bool 
Restrictor::isRestricted() const {
   bool tmp=true;
   for(size_t i=0;i<restrictons_.size();++i)
      tmp&=restrictons_[i]->isRestricted();
   return tmp;
}

inline bool 
Restrictor::canBeResticed() const {
   bool tmp=true;
   for(size_t i=0;i<restrictons_.size();++i)
      tmp&=restrictons_[i]->canBeResticed();
   return tmp;
}

inline size_t 
Restrictor::numberOfRestrictions() const {
   return restrictons_.size();
}

inline bool 
Restrictor::isRestricted(const size_t i) const {
   return restrictons_[i]->isRestricted();
}

inline bool 
Restrictor::canBeRestricted(const size_t i) const {
   return restrictons_[i]->canBeRestricted();
}

std::string 
Restrictor::name(const size_t i) const {
   return restrictons_[i]->name();
}

inline std::string 
Restrictor::description(const size_t i) const {
   return restrictons_[i]->description();
}

#endif	/* RESTICTOR_HXX */

