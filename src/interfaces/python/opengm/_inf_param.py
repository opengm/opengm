class _MetaInfParam(object):
   def __init__(self):
      pass

class InfParam(_MetaInfParam) :
   def __init__(self,*args,**kwargs):
      super(InfParam,self).__init__()
      self.args=args
      self.kwargs=kwargs