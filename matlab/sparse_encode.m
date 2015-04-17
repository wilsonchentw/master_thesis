function features = sparse_encode(train_list, test_list, dict_size)
    features = struct('train', zeros(dict_size, length(train_list)), ...
                      'test', zeros(dict_size, length(test_list)));
    param = struct('K', dict_size, 'lambda', 1, ...
                   'iter', 1000, 'verbose', false);

    train_vocabs = double([train_list.d]);
    test_vocabs = double([test_list.d]);
    dict = mexTrainDL_Memory(train_vocabs, param);

    for idx = 1:length(train_list)
        alpha = mexLasso(double(train_list(idx).d), dict, param);
        hist = mean(alpha, 2);    % Mean pooling
        features.train(:, idx) = hist;
    end

    for idx = 1:length(test_list)
        alpha = mexLasso(double(test_list(idx).d), dict, param);
        hist = mean(alpha, 2);    % Mean pooling
        features.test(:, idx) = hist;
    end
end

