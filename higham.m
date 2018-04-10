function [f] = higham(t, L)

  f = t;
  for j = 1:L
    f = sqrt(f);
  end
  for j = 1:L
    f = f**2;
  end
  f = f**2;
  return;
