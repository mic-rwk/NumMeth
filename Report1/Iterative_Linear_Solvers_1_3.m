%% PROBLEM 1: 4x4 System
disp('--- Problem 1 ---');
A1 = [ 2, -1,  0,  0;
      -1,  2, -1,  0;
       0, -1,  2, -1;
       0,  0, -1,  2];
b1 = [0; 0; 0; 5];
x0_1 = zeros(4,1); % Initial guess

% Direct method (Gaussian elimination)
tic; x_exact1 = A1 \ b1; t_gauss1 = toc;

% Iterative method (PCG)
tic;
[x_pcg1, flag1, relres1, iter1, resvec1] = pcg(A1, b1, 1e-6, 100, [], [], x0_1);
t_pcg1 = toc;

fprintf('Gaussian elimination time: %f s\n', t_gauss1);
fprintf('Iteration time (PCG): %f s, Iterations: %d\n', t_pcg1, iter1);


%% PROBLEM 3: System 2x2 with perturbation
disp('--- Problem 3 ---');
A3 = [0.835, 0.667;
      0.333, 0.266];
b3 = [0.168; 0.067];
b3_perturbed = [0.168; 0.066]; % Perturbed b vector

% Exact solution (Gauss)
x_exact3 = A3 \ b3;
fprintf('Exact solution (original b): [%.1f, %.1f]\n', x_exact3(1), x_exact3(2));

% Iterative solution for perturbed b (GMRES)
x0_3 = zeros(2,1);
[x_pert3, ~, ~, ~, resvec3] = gmres(A3, b3_perturbed, [], 1e-6, 100, [], [], x0_3);

fprintf('GMRES solution (perturbed b): [%.1f, %.1f]\n', x_pert3(1), x_pert3(2));
fprintf('Iterations: %d\n', length(resvec3)-1);