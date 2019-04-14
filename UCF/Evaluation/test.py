import numpy as np

from utils import interpolated_prec_rec


prec = np.array([0, 1, 0.5, 0.2, 0.1])
rec = np.array([0, 0.1, 0.5, 0.8, 1])
ap = interpolated_prec_rec(prec, rec)
print(ap)