// Gmsh project created on Wed May 31 16:13:13 2017
//+
Point(1) = {0, 1.22243671, 4e-16, 1.0};
//+
Point(2) = {-3.87870578e-1, 1.44755271, 6.91582087e-3, 1.0};
//+
Point(3) = {-2.96996101e-1, 1.10268759e0, -2.69393580e-2, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 1};
//+
Point(4) = {-5, -5, 1e-3, 1.0};
//+
Point(5) = {-5, 5, 1e-3, 1.0};
//+
Point(6) = {5, 5, 1e-3, 1.0};
//+
Point(7) = {5, -5, 1e-3, 1.0};
//+
Line(4) = {5, 4};
//+
Line(5) = {4, 7};
//+
Line(6) = {7, 6};
//+
Line(7) = {6, 5};
