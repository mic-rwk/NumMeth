%% Problem 4
% Effects of an ill-conditioned matrix with a small perturbation.
A = [0.835 0.667; 
     0.333 0.266];
b1 = [0.168; 0.067];
b2 = [0.168; 0.066]; 
cond_A = cond(A);
x1 = A \ b1;
x2 = A \ b2;
disp(['Condition number: ', num2str(cond_A)]);
disp('Solution for the original b1:'); 
disp(x1);
disp('Solution for the perturbed b2:'); 
disp(x2);

%% Problem 5
% Finding the inverse matrix by solving AX = I.
disp('--- Problem 5 ---');
A = [2 1 2; 
     1 2 3; 
     4 1 2];
I3 = eye(3);
A_inv = A \ I3;
disp('Inverse matrix A^-1 (calculated via A \ I3):');
disp(A_inv);

%% Problem 6
% LU factorization and calculating the determinant.
disp('--- Problem 6 ---');
A = [ 1  2  3  4; 
     -1  1  2  1; 
      0  2  1  3; 
      0  0  1  1];
b = [1; 1; 1; 1];
[L, U, P] = lu(A);
det_A = prod(diag(U)) * det(P);
y = L \ (P * b);
x = U \ y;
disp(['Calculated determinant: ', num2str(det_A)]);
disp('System solution:'); 
disp(x);