import numpy as np

pList = []
x_max = 0.4
x_min = 0.1
y_max = 0.4
y_min = -0.4
z_max = 0.2
z_min = 0.1
xlist = np.linspace(x_min, x_max, 30).tolist()
ylist = np.linspace(y_min, y_max, 30).tolist()
zlist = np.linspace(z_min, z_max, 30).tolist()
for x in xlist:
    for y in ylist:
        for z in zlist:
            r = np.sqrt(x**2+y**2+z**2)
            r_xy = np.sqrt(x**2+y**2)
            if (r < 0.7) and (0.1<r_xy) and ((x<0) or (0<x and 0.25<y) or (0<x and y<-0.25) or (0<x and -0.25<y and y<0.25 and 0.4<z)):
                pList.append([x,y,z])
                print(len(pList))
f = open('DynamicPath4/data/points.txt', 'w')
for p in pList:
    f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")
f.close()
