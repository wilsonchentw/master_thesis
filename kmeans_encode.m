function features = kmeans_encode(train_descriptor, test_descriptor, dict_size)
    % Generate codebook by K-means
    if isa([train_descriptor.d], 'uint8') && isa([test_descriptor.d], 'uint8')
        [dict, train_asgn] = vl_ikmeans([train_descriptor.d], ...
                                        dict_size, 'method','elkan');
        test_asgn = vl_ikmeanspush([test_descriptor.d], dict);
    else
        [dict, train_asgn] = vl_kmeans(double([train_descriptor.d]), ...
                                       dict_size, 'algorithm', 'elkan');
        [~, test_asgn] = min(vl_alldist2(dict, double([test_descriptor.d])));
    end

    % Encode testing image by codebook histogram
    train_encode = kmeans_hists(train_asgn, [train_descriptor.n], dict_size);
    test_encode = kmeans_hists(test_asgn, [test_descriptor.n], dict_size);

    % Normalize quantized SIFT descriptor histogram
    %train_encode = bsxfun(@rdivide, train_encode, sum(train_encode));
    %test_encode = bsxfun(@rdivide,test_encode, sum(test_encode));

    features.train = train_encode;
    features.test = test_encode;
end


