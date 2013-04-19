from _inference import *

class Minimizer:
   def neutral(self):
      return float("inf")

class Maximizer:
   def neutral(self):
      return float("-inf")






if __name__ == "__main__":
    import doctest
    doctest.testmod()