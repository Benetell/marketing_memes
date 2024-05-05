import numpy as np

x = np.array([83,70,0.3])
y = np.array([13,58,0.3])
z = np.array([17,66,0.7])

X = z - x
Y = y - x

cross = np.cross(X, Y)

print(cross[0]/cross[2])

x = np.array([8,12.3,9.1])
y = np.array([14.1,14.1,14.1])
print(np.cross(x,y))