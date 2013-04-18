class _MetaInfParam(object):
   def __init__(self):
      pass

class InfParam(_MetaInfParam) :
   def __init__(self,*args,**kwargs):
      super(InfParam,self).__init__()
      self.args=args
      self.kwargs=kwargs
      self._subInfMetaParam=self.__getSubInfMetaParam()

   """
   def hasSubInfMetaParam(self):
      return bool(self._subInfMetaParam is not None)
   def __getSubInfMetaParam(self):
      for param in self.args :
         if isinstance(param,InfParam):
            return param
      for argname in self.kwargs:
         param = self.kwargs[argname]
         if isinstance(param,InfParam):
            return param
      return None
   def replaceSubInfMetaParam(self,realParam):
      assert self.hasSubInfMetaParam()==True 
      for i,param in enumerate(self.args):
         if isinstance(param,InfParam):
            # replace
            self.args=tuple( aa  for aa in _tupleReplacerGen(self.args,i,realParam) )
            break
      for argname in self.kwargs:
         param = self.kwargs[argname]
         if isinstance(param,InfParam):
            del self.kwargs[argname]
            self.kwargs[argname]=realParam
            break
   """