function features = kmeans_encode(train_list, test_list, dict_size)
    % Generate codebook by K-means
    if isa([train_list.d], 'uint8') && isa([test_list.d], 'uint8')
        [dict, train_asgn] = vl_ikmeans([train_list.d], ...
                                        dict_size, 'method','elkan');
        test_asgn = vl_ikmeanspush([test_list.d], dict);
    else
        [dict, train_asgn] = vl_kmeans(double([train_list.d]), ...
                                       dict_size, 'algorithm', 'elkan');
        [~, test_asgn] = min(vl_alldist2(dict, double([test_list.d])));
    end

    % Encode testing image by codebook histogram
    train_encode = kmeans_hists(train_asgn, [train_list.n], dict_size);
    test_encode = kmeans_hists(test_asgn, [test_list.n], dict_size);

    % Normalize quantized SIFT descriptor histogram
    %train_encode = bsxfun(@rdivide, train_encode, sum(train_encode));
    %test_encode = bsxfun(@rdivide,test_encode, sum(test_encode));

    features.train = train_encode;
    features.test = test_encode;
end

function hists = kmeans_hists(assignment, num_descriptors, dict_size)
    hists = zeros(dict_size, length(num_descriptors));
    offset = cumsum(num_descriptors)-num_descriptors;
    for idx = 1:length(num_descriptors)
        v = assignment(offset(idx)+1:offset(idx)+num_descriptors(idx));
        hists(:, idx) = vl_ikmeanshist(dict_size, v);
    end
end


