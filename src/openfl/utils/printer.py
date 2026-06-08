import yaml
import sys
from datetime import datetime
from openfl.utils.config import get_print_config


# Ensure console output can handle unicode (e.g. arrows) on Windows, where the
# default console codec (cp1252) raises UnicodeEncodeError. Guard with hasattr:
# under some debuggers / pytest capture, stdout may not support reconfigure.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (ValueError, OSError):
            pass


config = get_print_config()

# PRITNS ONLY IF IT IS IN ENBALED_TAGES
ENABLED_TAGS = set(["setup_env"])

_log_file = None  # file handle set by set_log_file()

def set_enabled_tags(tags):
    global ENABLED_TAGS
    ENABLED_TAGS.update(tags)

def set_log_file(path: str):
    """Open a file for persistent logging. All log() calls also write there."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
    _log_file = open(path, "a", buffering=1, encoding="utf-8")  # line-buffered, utf-8 for unicode (e.g. arrows)

def log(tag, *args, **kwargs):
    if tag in ENABLED_TAGS:
        print(*args, **kwargs)
    if _log_file is not None:
        # Always write every tagged log line to file regardless of ENABLED_TAGS
        line = " ".join(str(a) for a in args)
        _log_file.write(f"[{tag}] {line}\n")


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