import yaml
import sys
from datetime import datetime
from openfl.utils.config import get_print_config


config = get_print_config()

# PRITNS ONLY IF IT IS IN ENBALED_TAGES
ENABLED_TAGS = set(["autorunner"])

def set_enabled_tags(tags):
    global ENABLED_TAGS
    ENABLED_TAGS.update(tags)

def log(tag, *args, **kwargs):
    if tag in ENABLED_TAGS:
        ts = datetime.now().strftime("[%m-%d %H:%M]")
        print(ts, *args, **kwargs)


def fmt_floats(values, precision=6, with_sum=True):
    body = "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"
    if with_sum:
        return f"{body}  (sum={sum(float(v) for v in values):.{precision}f})"
    return body


def fmt_scaled_scores(scores, scale=1e18, precision=6):
    # Display 1e18-scaled integer scores as readable floats.
    return fmt_floats([int(s) / scale for s in scores], precision=precision)

#print(config.ONLY_PRINT_ROUND_SUMMARY)
def _print(string, end= ""):
    if config.ONLY_PRINT_ROUND_SUMMARY:
        try:
            print(string.split(":")[0]+ string.split(":")[1].split("|")[0] +
                  "                                                              ", end = "\r")
        except:
            pass
        return
    print(string, end=end)

def print_bar(tag, i, l):
        if tag not in ENABLED_TAGS:
            return
        p = "-" * (i+1)
        r = "." *((l-1)-i)
        _print("{}{}".format(p, r), end="\r")