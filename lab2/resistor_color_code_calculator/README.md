Determine resistance, tolerance, and TCR of a resistor from its colored bands [[*]](#1).

### Usage
#### In the command prompt
```console
$ python color_code.py orange orange green gold

3.3MΩ ±5%
```
#### In a Python script
```py
from color_code import get_resistance, fmt_resistance

list_of_colors = ["orange", "orange", "green", "gold"]
result = get_resistance(list_of_colors)
print(result["resistance"]) # 3300000
print(fmt_resistance(**result)) # 3.3MΩ ±5%
```

<a id="1">[*]</a> https://en.wikipedia.org/wiki/Electronic_color_code#Color_band_system