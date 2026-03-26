from matplotlib import pyplot as plt
import numpy as np
from termcolor import colored


def get_color(i, a):
  if a == "bad":
      return bad_c
  if a == "freerider":
      return free_c
  try:
      return colors[i]
  except:
      return None
  
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bad_c  = "#d62728"
free_c = "#9467bd"
colors.remove(bad_c)
colors.remove(free_c)

def green(text):
    return colored(text, "green")

def gb(string):
    return colored(string, color="green", attrs=["bold"])

def rb(string):
    return colored(string, color="red", attrs=["bold"])

def b(string):
    return colored(string, color=None, attrs=["bold"])

def red(text):
    return colored(text, "red")

def yellow(text):
    return colored(text, "yellow", attrs=["bold"])

RNG = np.random.default_rng()