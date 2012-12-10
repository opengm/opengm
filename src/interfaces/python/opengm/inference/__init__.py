from _inference import *


class Minimizer:
   def neutral(self):
      return float("inf")

class Maximizer:
   def neutral(self):
      return float("-inf")


adder.minimizer.Accumulator = Minimizer
multiplier.minimizer.Accumulator = Minimizer
adder.minimizer.Accumulator = Maximizer
multiplier.minimizer.Accumulator = Maximizer


