
fx[0] =
- phys_x[0]
+ 0.25*x[0]*(-1 + x[0])*x[1]*(-1 + x[1])*vertices[0]
+ 0.25*x[0]*(1 + x[0])*x[1]*(-1 + x[1])*vertices[2]
+ 0.25*x[0]*(1 + x[0])*x[1]*(1 + x[1])*vertices[4];
+ 0.25*x[0]*(-1 + x[0])*x[1]*(1 + x[1])*vertices[6]
- 0.5*(1 + x[0])*(-1 + x[0])*x[1]*(-1 + x[1])*vertices[8]
- 0.5*x[0]*(1 + x[0])*(1 + x[1])*(-1 + x[1])*vertices[10]
- 0.5*(1 + x[0])*(-1 + x[0])*x[1]*(1 + x[1])*vertices[12]
- 0.5*x[0]*(-1 + x[0])*(1 + x[1])*(-1 + x[1])*vertices[14]
+ (1 + x[0])*(-1 + x[1])*(1 + x[1])*(-1 + x[1])*vertices[16]



fx[1] = -0.5*(1 + x[0])*(-1 + x[0])*x[1]*(-1 + x[1])*vertices[9] - 0.5*(1 + x[0])*(-1 + x[0])*x[1]*(1 + x[1])*vertices[13] + (1 + x[0])*(-1 + x[1])*(1 + x[1])*(-1 + x[1])*vertices[17] - phys_x[1] - 0.5*x[0]*(-1 + x[0])*(1 + x[1])*(-1 + x[1])*vertices[15] + 0.25*x[0]*(-1 + x[0])*x[1]*(-1 + x[1])*vertices[1] + 0.25*x[0]*(-1 + x[0])*x[1]*(1 + x[1])*vertices[7] - 0.5*x[0]*(1 + x[0])*(1 + x[1])*(-1 + x[1])*vertices[11] + 0.25*x[0]*(1 + x[0])*x[1]*(-1 + x[1])*vertices[3] + 0.25*x[0]*(1 + x[0])*x[1]*(1 + x[1])*vertices[5];