import json
import collections as cl

"""
{
    "itta_01_flowCroped": {
        "label": "itta"
    },
    "itta_02_flowCroped": {
        "label": "itta"
    },
    "itta_03_flowCroped": {
        "label": "itta"
    },
    "itta_04_flowCroped": {
        "label": "itta"
    },
    "itta_05_flowCroped": {
        "label": "itta"
    },
    "yanagiba_08_flowCroped": {
        "label": "yanagiba"
    },
    "yanagiba_09_flowCroped": {
        "label": "yanagiba"
    },
    "yanagiba_10_flowCroped": {
        "label": "yanagiba"
    }
}"""

annotations = {}
names = ["itta", "kuwabara", "sasaki", "sato", "sudo", "taiga", "toyoda", "tsubaki", "yamada", "yanagiba"]
ys = cl.OrderedDict()

for name in names:
	i = 1
	d = 1
	for i in range(1,31):
		i = str(i).zfill(2)
		n = int(i)
		name = str(name)
		annotations[name+"_"+i+"_cro"] = {"label": name, "num": n, "day": d}
		i = int(i)
		i += 1
		if 10 <= n <= 20:
			d = 2
		if 20 <= n <= 30:
			d = 3

print(annotations)
f = open("./data/annotations/day1-3_rgb.json", 'w')
json.dump(annotations, f, indent=2,)