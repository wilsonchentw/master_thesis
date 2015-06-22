function run_thesis(image_list)
    [prefix, label, path] = parse_list(image_list);
    folds = cross_validation(label, 5);

    % -------------------------------------------------------------------------
    %  Extract descriptors
    % -------------------------------------------------------------------------

    % Local binary pattern
    ds = extract_lbp(path);

    % Merge channel descriptor together
    ds = cellfun(@(x) {cell2mat(x)}, ds);   

    %% Split channel descriptor apart
    %ds = [ds{:}];

    % -------------------------------------------------------------------------
    %  Cross validation
    % -------------------------------------------------------------------------

    for cv = 1:length(folds)
        train_idx = folds(cv).train;
        test_idx = folds(cv).test;

        % Generate descriptor basis
        basis = generate_basis(ds(:, train_idx));

        % Generate feature vector
        feature = encode_descriptor(basis, ds);

        % Approximate kernel mapping
        feature = cell2mat(feature);
        feature = vl_homkermap(feature, 3, 'kernel', 'kinters', 'gamma', 1);

        % Linear classification
        train_inst = sparse(feature(:, train_idx));
        test_inst = sparse(feature(:, test_idx));
        train_label = double(label(train_idx));
        test_label = double(label(test_idx));

        % Generate model
        model = train(train_label, train_inst, '-s 1 -c 10 -q', 'col');

        % Generate top-N accuracy
        [g, acc, prob] = predict(test_label, test_inst, model, '', 'col');
        %rank_label = rank_candidate(test_inst, model);
        %acc = calculate_accuracy(test_label', rank_label);
    end
end


function basis = generate_basis(ds)
    num_basis = size(ds, 1);

    basis = cell(num_basis, 1);
    for ch = 1:num_basis
        vocabs = cell2mat(ds(ch, :));

        %% Hirarchical K-means codebook
        %branch = 2;
        %level = 10;
        %vocabs = uint8(round(vocabs * 255));
        %dict = kmeans_dict(vocabs, branch, level);
        %basis{ch} = double(dict) / 255.0;

        % Sparse coding basis
        param = struct('K', 1024 / 16, 'lambda', 0.25, 'lambda2', 0, ...
                        'iter', 400, 'mode', 2, 'modeD', 0, ...
                       'batchsize', 512, 'modeParam', 0, 'clean', true, ...
                       'numThreads', 4, 'verbose', false);
        basis{ch} = mexTrainDL(vocabs, param);
    end
end


function feature = encode_descriptor(dict, ds)
    num_channel = size(ds, 1);
    num_inst = size(ds, 2);

    feature = cell(num_channel, 1);
    for ch = 1:num_channel
        vocabs = ds(ch, :);

        %% Concatenate (CONCAT)
        %feature{ch} = reshape(cell2mat(vocabs), [], length(vocabs));

        %% Vector quantization (VQ)
        %feature{ch} = vq_encode(dict{ch}, vocabs);

        %% Locality-constrained linear coding (LLC)
        %feature{ch} = llc_encode(dict{ch}, vocabs);

        % Least absolute shrinkage and selection operator (LASSO)
        param = struct('lambda', 0.25, 'lambda2', 0, ...
                       'mode', 2, 'numThreads', 4);
        feature{ch} = lasso_encode(dict{ch}, vocabs, param);

        % Normalization
        feature{ch} = normalize_column(feature{ch}, 'L1');
    end
end
