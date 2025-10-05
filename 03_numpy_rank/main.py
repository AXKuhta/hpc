import matplotlib.pyplot as plt
import numpy as np

# (Probably) a full rank matrix
fullrank = np.random.rand(10, 10)

# Lower rank matrix
lowrank = fullrank.copy()
lowrank[5] = 4*lowrank[1] + lowrank[3]
lowrank[8] = 4*lowrank[4] + 8*lowrank[6]
lowrank[2] = 3*lowrank[7]

# Answers
print("Full rank:", np.linalg.matrix_rank(fullrank))
print("Low rank:", np.linalg.matrix_rank(lowrank))
