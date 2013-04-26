

def _injectGenericInferenceParameterInterface(solverParamClass,infParam,subInfParam):
   BoostPythonMetaclass=solverParamClass.__class__
   class InjectorGenericInferenceParameter(object):
      def __init__(self,*args,**kwargs):
         print "in init"
      class __metaclass__(BoostPythonMetaclass):
         def __init__(self, name, bases, dict):
            for b in bases:
               if type(b) not in (self, type):
                  for k,v in dict.items():
                     #print "attr: key= ",k
                     setattr(b,k,v)


               if( issubclass(b ,solverParamClass)):
                  baseInit=b.__init__
                  #baseAttr=b.__setattr__
                  #baseSetAttr=b.__setattr__
                  #setattr(b,'_orginal_setattr__',baseSetAttr)

                  def newInit(*args,**kwargs):
                     self_proxy=args[0]
                     assert isinstance(self_proxy,solverParamClass)
                     baseInit(self_proxy)
                     self_proxy._after_init(*args[1:len(args)],**kwargs)

                  def newSetAttr(*args,**kwargs):
                     self_proxy=args[0]
                     assert isinstance(self_proxy,solverParamClass)
                     name,nativeValue=self_proxy._setattr_helper__(*args[1:len(args)],**kwargs)
                     print nativeValue
                     super(solverParamClass,self_proxy).__setattr__(name, nativeValue)

                  b.__setattr__=newSetAttr   
                  b.__init__=newInit
            return type.__init__(self, name, bases, dict)


   class PyAddon_GenericInferenceParameter(InjectorGenericInferenceParameter,solverParamClass):
      
      #def _bevore_init(self,*args,**kwargs):
         #print "bevore init",
      def _after_init(self,*args,**kwargs):
         self.set()
         if( len(args)==0 and len(kwargs)==0):
            self.set()
         elif(len(kwargs)==0):
            if(len(args)==1):
               argValue=args[0]
               argValueClass=argValue.__class__
               if(is_meta_inf_param(classType=argValueClass)):
                  for pName in argValue.kwargs.keys():
                     setattr(self,pName,argValue.kwargs[pName])

      def _setattr_helper__(self, name, value):
        if hasattr(self, name):
            # get target class
            targetClass=getattr(self,name).__class__
            # convert class to native representation
            return name,to_native_class_converter(givenValue=value,nativeClass=targetClass)
        else:
            raise NameError("%s has no attribute named \'%s\' , the parameter attributes are : %s " % (str(solverParamClass),str(name),str(self.getParameterNames()) ))

      @staticmethod
      def _is_inf_param( ):
         return bool(infParam)
      @staticmethod
      def _is_sub_inf_param( ):
         return bool(subInfParam)


      def parameterNames(self):
         """ returns a generator object to iterate over the names of the parameter
         """
         for propertyName, value in vars(self.__class__).iteritems(): 
            if( (propertyName.startswith('__') or propertyName.endswith('__')) ==False ):
               #check if it is a property
               if( repr(value).find('property')!=-1):
                 yield propertyName

      def parameterNamesAndTypes(self):
         """ returns a generator object to iterate over the names and types of the parameter"""
         for name in self.parameterNames():
            yield name,getattr(self,name).__class__
      def numberOfAttributes(self):
         """ returns the number of attributes of this parameter object
         """
         c=0
         for propertyName in self.parameterNames():
            c+=1
         return c
      def getParameterNames(self):
         """ returns a list with the parameter names"""
         return [propertyName for propertyName in self.parameterNames() ]



      def _attributeTypeDict(self):
         adict=dict()
         for propertyName, value in vars(self.__class__).iteritems(): 
            if( (propertyName.startswith('__') or propertyName.endswith('__')) ==False ):
               #check if it is a property
               if( repr(value).find('property')!=-1):
                  attrValue=getattr(self, propertyName)
                  classOfAttr=attrValue.__class__
                  if is_sub_inf_param(classType=classOfAttr) :
                     adict[propertyName]=(classOfAttr,True)
                  else:
                     adict[propertyName]=(classOfAttr,False)
         return adict



      def __str__(self):
         old_stdout = sys.stdout
         sys.stdout = mystdout = StringIO()
         c=0
         for propertyName, value in vars(solverParamClass).iteritems(): 
            if( (propertyName.startswith('__') or propertyName.endswith('__')) ==False ):
               #check if it is a property
               if( repr(value).find('property')!=-1):
                  attrValue=getattr(self, propertyName)
                  if c>0:
                     sys.stdout.write(", ")
                  c+=1
                  sys.stdout.write(str(propertyName))
                  sys.stdout.write("=")
                  sys.stdout.write(str(attrValue))
                  sys.stdout.write("\n\n\n")
         sys.stdout = old_stdout
         return  mystdout.getvalue()
      def _str_spaced_(self,space=None):
         if space is None:
            space='         '
         old_stdout = sys.stdout
         sys.stdout = mystdout = StringIO()
         print space+"Parameters of this Solver:\n\n"
         for propertyName, value in vars(solverParamClass).iteritems(): 
            if( (propertyName.startswith('__') or propertyName.endswith('__')) ==False ):
               #check if it is a property
               if( repr(value).find('property')!=-1):
                  attrValue=getattr(self, propertyName)
                  sys.stdout.write(space)
                  sys.stdout.write(str("* "))
                  sys.stdout.write(str(propertyName))
                  sys.stdout.write("=")
                  if(propertyName=='subInfParam'):
                     sys.stdout.write(str("\n\n\n"))
                     sys.stdout.write(str(attrValue._str_spaced_(space+'   ')))
                  else:
                     sys.stdout.write(str(attrValue))
                     sys.stdout.write("\n\n\n")
                     sys.stdout.write(space+'   ')

                     docStrAttr=str(solverParamClass.__dict__[propertyName].__doc__ )
                     sys.stdout.write(docStrAttr.replace("\n","\n"+space+space+space))
                  sys.stdout.write("\n\n\n")
         sys.stdout = old_stdout
         return  mystdout.getvalue()
