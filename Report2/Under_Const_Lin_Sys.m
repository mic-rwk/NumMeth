% Constrained Linear Systems
% Problem 6: NNLS Comparison
A6 = [73, 71, 52; 87, 74, 46; 72, 2, 7; 80, 89, 71]; 
b6 = [49; 67; 68; 20];

% Solutions
x_man = nnls(A6, b6); % Replace 'nnls' with your own custom function if you have one
x_mat = lsqnonneg(A6, b6);

fprintf("Problem 6\n");
fprintf('Solution (Manual):\n');
disp(x_man);
fprintf('Solution (Matlab):\n');
disp(x_mat);
fprintf('Residual (Manual): %.4f\n', norm(b6 - A6*x_man));
fprintf('Residual (Matlab): %.4f\n\n', norm(b6 - A6*x_mat));

% Problem 7: NNLS vs LS with Noise
A7 = [-4, -2, -4, -2; 2, -2, 2, 1; -4, 1, -4, -2]; 
b7 = [-12; 3; -9];

% Ideal Data
x_nnls_ideal = nnls(A7, b7);
x_ls_ideal = A7 \ b7;

fprintf('Problem 7\n');
fprintf('--- Ideal Data ---\n');
fprintf('NNLS Solution:\n');
disp(x_nnls_ideal);
fprintf('LS Solution:\n');
disp(x_ls_ideal);
fprintf('Ideal || NNLS Res: %.4f, LS Res: %.4f\n\n', norm(b7 - A7*x_nnls_ideal), norm(b7 - A7*x_ls_ideal));

% Noisy Data (SNR = 20dB)
noise = sqrt((norm(b7)^2 / length(b7)) / 10^(20/10)) * randn(size(b7));
b_noisy = b7 + noise;

x_nnls_noisy = nnls(A7, b_noisy);
x_ls_noisy = A7 \ b_noisy;

fprintf('--- Noisy Data (SNR = 20dB) ---\n');
fprintf('NNLS Solution:\n');
disp(x_nnls_noisy);
fprintf('LS Solution:\n');
disp(x_ls_noisy);
fprintf('Noisy || NNLS Res: %.4f, LS Res: %.4f\n', norm(b_noisy - A7*x_nnls_noisy), norm(b_noisy - A7*x_ls_noisy));