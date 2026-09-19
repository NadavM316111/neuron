"""Rung 4: a stream nobody designed.

Rungs 1 through 3 were all worlds we wrote the rules for. This is real
data: hourly weather readings from an actual station, four years of them.
Nobody chose the dynamics, nobody balanced the classes, and there is no
hidden variable anyone can hand the model.

That last point changes the method. Every experiment so far had a CEILING
arm that could read the hidden state, which made "how much is the memory
costing" a clean subtraction. Here nobody knows what the hidden state is,
so the comparison has to be against baselines instead:

  persistence     predict no change from the current reading. For weather
                  this is a genuinely strong baseline and beating it is not
                  automatic.
  extrapolation   continue the last change. Stronger still on smooth
                  series.
  offline         the same network trained conventionally, shuffled, many
                  passes. What a training run would achieve.

THE TASK. Given the current readings only, predict what the temperature
will do over the next three hours: rise, hold, or fall.

Deliberately NO clock and NO calendar in the input. The model gets
temperature, humidity, pressure and wind, and nothing else. To do well it
has to infer time of day and season from the pattern of what it has seen,
which is exactly the kind of hidden state the grid worlds tested with keys
and lamps, except here nobody planted it.

Three hours ahead rather than one, because one hour ahead is nearly always
"hold" and persistence would win by default. This file reports the class
balance so you can see whether that worked.
"""

import json
import math
import os
import urllib.request

CACHE = "weather.json"

# Somewhere with strong seasons and real weather, so the stream is
# genuinely non-stationary rather than a flat line with noise.
LATITUDE = 41.88          # Chicago
LONGITUDE = -87.63
START = "2019-01-01"
END = "2023-12-31"

VARIABLES = ["temperature_2m", "relative_humidity_2m",
             "surface_pressure", "wind_speed_10m"]

HORIZON = 3               # hours ahead to predict
RISE = 1.0                # degrees C that counts as a real change
CLASSES = ["falling", "holding", "rising"]

# Coordinates for the climates used across experiments. replicate.py passed
# these directly; fetch() only knew one location and took `force` as its
# first argument, so calling fetch("reykjavik") set force=True and
# re-downloaded the SAME city into the SAME cache file. Three "different"
# climates came back byte-identical.
CLIMATES = {
    "chicago": (41.88, -87.63),
    "singapore": (1.35, 103.82),
    "phoenix": (33.45, -112.07),
    "reykjavik": (64.15, -21.94),
    "darwin": (-12.46, 130.84),
}


def fetch(climate=None, force=False):
    """Hourly readings from Open-Meteo's archive. Free, no key needed.

    Cached PER CLIMATE. A single shared cache file meant every location
    overwrote the last one, which silently made cross-climate experiments
    compare a city to itself.
    """
    lat, lon = (CLIMATES[climate] if climate else (LATITUDE, LONGITUDE))
    cache = f"weather_{climate}.json" if climate else CACHE

    if os.path.exists(cache) and not force:
        with open(cache) as f:
            return json.load(f)

    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={START}&end_date={END}"
           f"&hourly={','.join(VARIABLES)}")
    print(f"fetching {climate or 'default'} ({lat}, {lon}), "
          f"{START} to {END} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "neuron"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    with open(cache, "w") as f:
        json.dump(data, f)
    print(f"cached to {cache}")
    return data


def build_stream(data):
    """Turn the raw readings into (features, label) moments.

    Features are the four current readings, standardised. NO time, NO date.
    The label is what temperature does over the next HORIZON hours.
    """
    h = data["hourly"]
    cols = [h[v] for v in VARIABLES]
    n = len(h["time"])

    # Drop any hour with a missing reading rather than interpolating, so
    # nothing is invented.
    keep = [i for i in range(n)
            if all(c[i] is not None for c in cols)
            and i + HORIZON < n
            and cols[0][i + HORIZON] is not None]

    # Standardise each variable over the whole record. A real deployment
    # could not do this, but the alternative is a running normaliser, which
    # is a separate design question and would confound this test.
    stats = []
    for c in cols:
        vals = [c[i] for i in keep]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
        stats.append((mu, sd))

    stream = []
    temps = cols[0]
    for i in keep:
        feats = [(cols[j][i] - stats[j][0]) / stats[j][1]
                 for j in range(len(cols))]
        change = temps[i + HORIZON] - temps[i]
        if change > RISE:
            label = "rising"
        elif change < -RISE:
            label = "falling"
        else:
            label = "holding"
        stream.append((feats, label, h["time"][i]))
    return stream


def baselines(stream):
    """What the simple predictors score, and what guessing scores.

    persistence   always "holding"
    extrapolation continue the last hour's direction
    majority      the most common class
    """
    counts = {}
    for _, lab, _ in stream:
        counts[lab] = counts.get(lab, 0) + 1
    total = len(stream)
    majority = max(counts.values()) / total * 100.0

    hold = 100.0 * counts.get("holding", 0) / total

    # extrapolation: the temperature feature is index 0, standardised
    right = 0
    for i in range(1, len(stream)):
        delta = stream[i][0][0] - stream[i - 1][0][0]
        guess = ("rising" if delta > 0.02 else
                 "falling" if delta < -0.02 else "holding")
        if guess == stream[i][1]:
            right += 1
    extrap = 100.0 * right / (len(stream) - 1)

    return dict(counts=counts, majority=majority, persistence=hold,
                extrapolation=extrap)


if __name__ == "__main__":
    data = fetch()
    stream = build_stream(data)

    print(f"\n{len(stream)} hourly moments, "
          f"{stream[0][2]} to {stream[-1][2]}")
    print(f"features: {', '.join(VARIABLES)}")
    print(f"no clock, no calendar. Time of day and season must be inferred.")
    print(f"task: what temperature does {HORIZON} hours ahead, "
          f"threshold {RISE} degrees\n")

    b = baselines(stream)
    print("class balance:")
    for c in CLASSES:
        n = b["counts"].get(c, 0)
        bar = "#" * int(50 * n / len(stream))
        print(f"  {c:>8}: {n:>6} ({100.0 * n / len(stream):5.2f}%) {bar}")

    print(f"\nbaselines to beat:")
    print(f"  majority class      {b['majority']:5.1f}%")
    print(f"  persistence         {b['persistence']:5.1f}%  "
          f"(always predict 'holding')")
    print(f"  extrapolation       {b['extrapolation']:5.1f}%  "
          f"(continue the last hour's direction)")

    print("\nnon-stationarity, mean standardised temperature by year:")
    by_year = {}
    for feats, lab, t in stream:
        y = t[:4]
        by_year.setdefault(y, []).append(feats[0])
    for y in sorted(by_year):
        vals = by_year[y]
        print(f"  {y}: {sum(vals) / len(vals):+6.3f}  ({len(vals)} hours)")

    print("\nby month, first year, to show the seasonal swing:")
    by_month = {}
    for feats, lab, t in stream:
        if t[:4] == sorted(by_year)[0]:
            by_month.setdefault(t[:7], []).append(feats[0])
    for m in sorted(by_month):
        vals = by_month[m]
        avg = sum(vals) / len(vals)
        bar = "#" * int(20 + 20 * avg)
        print(f"  {m}: {avg:+6.3f} {bar}")

    strongest = max(b["majority"], b["extrapolation"])
    print(f"""
The strongest simple baseline is {strongest:.1f}%. That is the number the
learning system has to beat for this rung to mean anything.

  balance roughly even across three classes -> good, there is something to
      predict and guessing will not carry it
  one class above 70% -> the task is too easy to guess and the threshold or
      the horizon needs adjusting before running anything
  extrapolation very high -> the series is smooth enough that continuing
      the last change is most of the answer, and again the horizon needs
      pushing out

The monthly swing is the non-stationarity. Unlike every world so far,
nobody put it there.
""")