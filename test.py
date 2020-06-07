import time
import enlighten

bar_format = (
    u"{desc}{desc_pad} Current loss:{i} {percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} "
    + u"[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
)

my_fields = {"i": 0}
pbar = enlighten.Counter(total=100, desc="Basic", unit="ticks", bar_format=bar_format, additional_fields=my_fields)

for num in range(100):
    my_fields["i"] += 1
    time.sleep(0.1)  # Simulate work
    pbar.update()
