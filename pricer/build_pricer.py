"""Build pricer/rift_pricer.html: inlines the maths and ratings.json (made by export_ratings.py) into one page.
Run from the repo root:  python export_ratings.py  then  python pricer/build_pricer.py"""
import json, os
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
rd = lambda p: open(p, encoding="utf-8").read()
t = rd(os.path.join(here, "pricer_template.html"))
math = rd(os.path.join(here, "pricer_math.js")).replace('if (typeof module !== "undefined") module.exports = RP;', "")
data = json.dumps(json.load(open(os.path.join(root, "ratings.json"), encoding="utf-8")), separators=(",", ":"))
out = os.path.join(here, "rift_pricer.html")
open(out, "w", encoding="utf-8").write(t.replace("/*__MATH__*/", math).replace("/*__DATA__*/", data))
print("built", out, len(rd(out)) // 1024, "KB")
