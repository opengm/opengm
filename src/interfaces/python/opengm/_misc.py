def defaultAccumulator(gm=None,operator=None):
   """
   return the default accumulator for a given gm or operator
   """
   if gm is not None:
      operator=gm.operator
   elif operator is None and gm is None:
      raise NameError("at least a gm or an operator must be given")
   if operator=='adder':
      return 'minimizer'
   elif operator=='multiplier':
      return 'maximizer'
   else:
      raise RuntimeError("unknown operator: "+ operator)




