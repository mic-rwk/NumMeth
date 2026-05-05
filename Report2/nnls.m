function x = nnls(A, b)
    n = size(A, 2); x = zeros(n, 1);
    P = false(n, 1); Z = true(n, 1);
    w = A' * (b - A * x);
    
    while any(Z & w > 1e-12)
        [~, t] = max(w .* Z);
        P(t) = true; Z(t) = false;
        z = zeros(n, 1); z(P) = A(:, P) \ b;
        
        while any(z(P) <= 0)
            Q = (z <= 0) & P;
            alpha = min(x(Q) ./ (x(Q) - z(Q)));
            x = x + alpha * (z - x);
            Z = Z | (x <= 1e-12); P = ~Z; x(Z) = 0;
            z(P) = A(:, P) \ b;
        end
        x = z; w = A' * (b - A * x);
    end
end