from _inf_param import _MetaInfParam, InfParam

def is_inf_param(classType=None,instanceType=None):
   try:
      if classType is not None:
         return classType._is_inf_param()
      else:
         return instanceType._is_inf_param()
   except:
      return False

def is_sub_inf_param(classType=None,instanceType=None):
   try:
      if classType is not None:
         return classType._is_sub_inf_param()
      else:
         return instanceType._is_sub_inf_param()
   except:
      return False



def is_meta_inf_param(classType=None,instanceType=None):
   if classType is None:
      classType = instanceType.__class__
   return issubclass(classType,_MetaInfParam)


def is_boost_python_enum(classType=None,instanceType=None):
   if classType is None:
      classType=instanceType.__class__ 
   try :
      classDict=classType.__dict__
      if 'names' in classDict:
         names=classDict['names']
         if(isinstance(names,dict)):
            # TODO: replace type check : check if type is the same as for an actual boost python enum
            if(len(names)>0 and type(classType)==type):
               return True
      return False
   except:
      return False


def is_build_in_simple_parameter(classType=None,instanceType=None):
   simple_types=[int,long,float,bool,str]
   if(instanceType is not None):
      for st in simple_types:
         if isinstance(instanceType,st) :
            return True
      return False
   else:
      for st in simple_types:
         if isinstance(st(),classType) :
            return True
      return False


def is_tribool(classType=None,instanceType=None):
   if classType is None:
      classType=instanceType.__class__
   return ('Tribool' in str(classType) or 'tribool' in str(classType) )



def is_string(classType=None,instanceType=None):
   if(instanceType is not None):
      return isinstance(instanceType,str)
   else:
      return isinstance(str(),classType)

def is_1d_seq_like(classType=None,instanceType=None):
    raise TypeError(' TODO!!!!!!!!!!!!!!!!!!!')



# special string detectors

# string as: '0.4,foo=1,bar='bar',fooBar=True '
def is_kwarg_arg_style_string():
    raise TypeError(' TODO!!!!!!!!!!!!!!!!!!!')



# comparators

def same_class(classA=None,classB=None,instanceA=None,instanceB=None):
   if classA is None : 
      classA=instanceA.__class__
   if classB is None : 
      classB=instanceB.__class__

   return issubclass(classA,classB) and issubclass(classB,classA)



class ContainerConvertPolicy():
    def __init__(self,fixedTypes=None,forceSize=None):
        self.fixedTypes=fixedTypes
        self.forceSize=forceSize




# converters

def to_native_boost_python_enum_converter(givenValue,nativeClass):
   if is_boost_python_enum(instanceType=givenValue) or is_string(instanceType=givenValue):
      givenStrName=str(givenValue).lower()
      # try to find givenStrName in name dict of other
      for nativeName in nativeClass.names:
         if givenStrName == nativeName.lower():
            return nativeClass.names[nativeName]
      raise TypeError('Cannot find the given name \'%s\' in the native enums names %s' % ( str(givenValue), str(nativeClass.names.keys()) ) )
   elif isinstance( givenValue,(int,long,float)):
      asInt=int(givenValue)
      if asInt in nativeClass.values :
         return nativeClass.values[asInt]
      else:
         raise TypeError('Cannot find the given value \'%i\' in the native enums values %s' % ( asInt, str(nativeClass.values) ) )
         
   elif same_class(classA=nativeClass,instanceB=givenValue):
      return givenValue
   else:
      raise TypeError('Cannot convert given value \'%s\' of the type \'%s\' to an boost python enum'%(str(givenValue),str(givenValue.__class__) ) )


def to_native_build_in_simple_class_converter(givenValue,nativeClass):
   # givenValue == a type as float int long str 
   if is_build_in_simple_parameter(instanceType=givenValue):
      return nativeClass(givenValue)
   else:
      raise TypeError('Cannot convert the class \'%s\' to the native class \'%s\'' % ( str(givenValue.__class__), str(nativeClass) ) )


def to_native_tribool_converter(givenValue,nativeClass):
   if isinstance(givenValue,(bool,int )):
      return nativeClass(givenValue)
   elif isinstance(givenValue,str) or is_boost_python_enum(givenValue):
      lstr=str(givenValue).lower()
      if lstr == 'true':
         return nativeClass(True)
      elif lstr == 'false':
         return nativeClass(False)
      elif lstr == 'maybe':
         return nativeClass(-1)
   elif same_class(classA=nativeClass,instanceB=givenValue):
      return givenValue
   else:
      raise TypeError('Cannot find the given name \'%s\' into  %s' % ( str(givenValue), str(['True','False','Maybe']) ) )




def to_native_inf_param_converter(givenValue,nativeClass):

   # nativeClass == the same as givenValue
   if same_class(classA=nativeClass,instanceB=givenValue):
      return givenValue
   else:
      cppParameter=nativeClass()
      cppParameter.set()
      if is_meta_inf_param(instanceType=givenValue):
         givenKwargs=givenValue.kwargs
         allowedKwargs=dict()
         for name,nativeType in cppParameter.parameterNamesAndTypes():
            allowedKwargs[name]=nativeType
         for name in givenKwargs:
            if name in allowedKwargs:
               setattr( cppParameter, name,givenKwargs[name])
            else:
               raise TypeError("Inference Parameter %s has no attribute %s" % (str(nativeClass),name ))
      else :
         raise TypeError("Inference Parameter %s cannot be constructed from  %s" % (str(givenValue.__class__),name ))
      return cppParameter






def to_native_class_converter(givenValue,nativeClass,converterPolicy=None):

   # nativeClass == the same as givenValue
   if same_class(classA=nativeClass,instanceB=givenValue):
      return givenValue

   # nativeClass == a type as float int long str 
   elif is_build_in_simple_parameter(classType=nativeClass):
      return to_native_build_in_simple_class_converter(givenValue=givenValue,nativeClass=nativeClass)

   # nativeClass == a boost python enum
   elif is_boost_python_enum(classType=nativeClass):
      return to_native_boost_python_enum_converter(givenValue=givenValue,nativeClass=nativeClass)


   # nativeClass == a tribool
   elif is_tribool(classType=nativeClass):
      return  to_native_tribool_converter(givenValue=givenValue,nativeClass=nativeClass)

   # nativeClass == a is_inf_param
   elif is_inf_param(classType=nativeClass) or is_sub_inf_param(classType=nativeClass):
      return to_native_inf_param_converter(givenValue=givenValue,nativeClass=nativeClass)

   # givenValue == tuple
   elif isinstance(givenValue,tuple):
      return to_native_inf_param_converter(givenValue=givenValue,nativeClass=nativeClass)

   # givenValue == list
   elif isinstance(givenValue,tuple):
      return to_native_inf_param_converter(givenValue=givenValue,nativeClass=nativeClass)

   else :
      raise RuntimeError(str(nativeClass)+" is unknown")
   # givenValue == string value
