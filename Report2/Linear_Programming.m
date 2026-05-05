% Linear Programming

% Problem 1
c1 = [1; 1]; 
A1 = [1, 2; 4, 2; -1, 1]; 
b1 = [4; 12; 1];
[x1, fval1, ~] = simplex(c1, A1, b1);
fprintf('Prob 1 \nx1: %.2f, x2: %.2f, Max: %.2f\n', x1(1), x1(2), fval1);

% Problem 2

c2 = [60; 120]; 
A2 = [2, 2; 1, 3]; 
b2 = [3; 2];
[~, fval2, x2] = simplex(c2, A2, b2);
fprintf('Prob 2 \nCheese: %.2f, Bread: %.2f, Min Cost: %.2f\n', x2(1), x2(2), fval2);