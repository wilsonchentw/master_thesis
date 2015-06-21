function lasso = lasso_encode(dict, vocabs, param)
    dict_size = size(dict, 2);

    lasso = zeros(dict_size, length(vocabs));
    for idx = 1:length(vocabs)
        alpha = mexLasso(vocabs{idx}, dict, param);

        % Mean pooling
        lasso(:, idx) = mean(alpha, 2);

        %% Max pooling
        %lasso(:, idx) = max(alpha, [], 2);
    end
end
