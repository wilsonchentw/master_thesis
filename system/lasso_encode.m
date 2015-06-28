function lasso = lasso_encode(dict, vocabs, param)
    dict_size = size(dict, 2);
    param = struct('lambda', 0.25, 'lambda2', 0, 'mode', 2, 'numThreads', 4);

    lasso = zeros(dict_size, length(vocabs));
    for idx = 1:length(vocabs)
        alpha = mexLasso(vocabs{idx}, dict, param);
        lasso(:, idx) = mean(alpha, 2);
    end
end
