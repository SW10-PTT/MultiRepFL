import yaml
import sys
from openfl.utils.config import get_print_config


config = get_print_config()

# PRITNS ONLY IF IT IS IN ENBALED_TAGES
ENABLED_TAGS = set(["autorunner"])

def set_enabled_tags(tags):
    global ENABLED_TAGS
    ENABLED_TAGS.update(tags)

def log(tag, *args, **kwargs):
    if tag in ENABLED_TAGS:
        print(*args, **kwargs)

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