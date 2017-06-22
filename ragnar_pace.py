import math
import sys

def average_pace(time):
    global counter
    time.split(':')
    hours, minutes, seconds = [float(item) for item in time.split(':')]
    minutes = hours * 60. + minutes + seconds / 60.
    pace_dec = minutes / 200.
    seconds, minutes = math.modf(pace_dec)
    seconds = seconds * 60
    print('{}: {:.0f}:{:02.0f}'.format(counter, minutes, seconds))
    counter += 1

try:
    counter = int(sys.argv[1])
    times = sys.argv[2:]
    for time in times:
        average_pace(time)
except:
    raise RuntimeError("You must supply at least two arguments to this script: "
                       "first: starting place of times, "
                       "second: the time in hours:minutes:seconds")
