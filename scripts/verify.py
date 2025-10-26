#!/usr/bin/env python3

import numpy as np
import ast
import sys

data = sys.stdin.read()

try:
    aa, bb = data.split("---SPLIT---")
except:
    print("ERROR: expected delimiter.", file=sys.stderr)

a = np.array(ast.literal_eval(aa))
b = np.array(ast.literal_eval(bb))

# Compare
if np.allclose(a, b, rtol=1e-4, atol=1e-4):
    sys.exit(0)
else:
    sys.exit(1)
