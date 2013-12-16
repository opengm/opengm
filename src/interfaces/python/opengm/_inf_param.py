class _MetaInfParam(object):
   def __init__(self):
      pass

class InfParam(_MetaInfParam) :
   def __init__(self,*args,**kwargs):
      super(InfParam,self).__init__()
      if len(args)!=0:
        raise RuntimeError("Inference parameter does only suppoty keyword arguments")
      #self.args=args
      self.kwargs=kwargs




