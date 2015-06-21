function llc = llc_encode(dict, vocabs)
    dict_size = size(dict, 2);

    llc = zeros(dict_size, length(vocabs));
    for idx = 1:length(vocabs)

        %% Exactly solution of LLC
        %sigma = 1.0;
        %lambda = 1.0;
        %alpha = llc_exact(dict, vocabs{idx}, sigma, lambda);

        % Approximate solution of LLC
        knn = 5;
        alpha = llc_approx(dict, vocabs{idx}, knn);

        % Mean pooling
        llc(:, idx) = mean(alpha, 2);
    end
end


function llc = llc_exact(B, X, sigma, lambda)
    x_num = size(X, 2);
    b_num = size(B, 2);

    llc = zeros(b_num, x_num);
    for idx = 1:size(X, 2)
        xi = X(:, idx);
        di = exp(vl_alldist2(B, xi, 'L2') / sigma);
        di = di / max(di);

        z = B - repmat(xi, 1, b_num);               % Shift properties
        Ci = z' * z;                                % Local covariance
        Ci = Ci + eye(b_num) * trace(Ci) * 1e-4;    % Regularization
        ci = (Ci + lambda * diag(di .* di)) \ ones(b_num, 1); 
        llc(:, idx) = ci / sum(ci);
    end
end


function llc = llc_approx(B, X, knn)
    x_num = size(X, 2);
    b_num = size(B, 2);

    kd_tree = vl_kdtreebuild(B);
    nearest_neighbors = vl_kdtreequery(kd_tree, B, X, 'NumNeighbors', knn);
    llc = zeros(b_num, x_num);
    for idx = 1:size(X, 2)
        xi = X(:, idx);
        nn = nearest_neighbors(:, idx);

        z = B(:, nn) - repmat(xi, 1, knn);   % Shift properties
        C = z' * z;                          % Local covariance
        C = C + eye(knn) * trace(C) * 1e-4;  % Regularization
        w = C \ ones(knn, 1);

        llc(nn, idx) = w / sum(w);
    end
end
