function [x, fval, x_dual] = simplex(c, A, b)
    [m, n] = size(A);
    tab = [A, eye(m), b; -c', zeros(1, m+1)];
    
    while any(tab(end, 1:end-1) < 0)
        [~, p_col] = min(tab(end, 1:end-1));
        ratios = inf(m, 1);
        idx = tab(1:m, p_col) > 0;
        ratios(idx) = tab(idx, end) ./ tab(idx, p_col);
        [~, p_row] = min(ratios);
        
        tab(p_row, :) = tab(p_row, :) / tab(p_row, p_col);
        for i = 1:size(tab, 1)
            if i ~= p_row, tab(i, :) = tab(i, :) - tab(i, p_col) * tab(p_row, :); end
        end
    end
    
    x = zeros(n, 1);
    for j = 1:n
        col = tab(1:m, j);
        if sum(col == 1) == 1 && sum(col == 0) == m - 1
            x(j) = tab(col == 1, end);
        end
    end
    fval = tab(end, end);
    x_dual = tab(end, n+1:end-1)';
end