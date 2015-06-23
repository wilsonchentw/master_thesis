function run_thesis(image_list)
    [prefix, label, path] = parse_list(image_list);
    folds = cross_validation(label, 5);

tic
    fprintf('Extract LAB-LBP\n');
    lbp = extract_lbp(path);
toc

    for cv = 1:length(folds)
        train_idx = folds(cv).train;
        test_idx = folds(cv).test;

        % Augmented data by flip LBP
        ds = [lbp{1, :}, lbp{2, train_idx}];
        aug_label = [label; label(train_idx)];
        aug_train_idx = setdiff(1:size(ds, 2), test_idx)';
        %aug_train_idx = train_idx;      % Delete augmented training data

        % Generate descriptor basis & feature vector
tic
        fprintf('\n\nGenerate descriptor basis\n');
        basis = generate_basis(ds(:, aug_train_idx));
toc
tic
        fprintf('\nGenerate feature vector\n');
        feature = encode_descriptor(basis, ds);
toc

tic
        fprintf('\nKernel mapping, classification, prediction\n');
        % Approximate kernel mapping
        feature = cell2mat(feature);
        feature = vl_homkermap(feature, 3, 'kernel', 'kinters', 'gamma', 1);

        % Linear classification
        train_inst = sparse(feature(:, aug_train_idx));
        train_label = double(aug_label(aug_train_idx));
        test_inst = sparse(feature(:, test_idx));
        test_label = double(aug_label(test_idx));

        % Generate model & top-N accuracy
        model = train(train_label, train_inst, '-s 1 -c 10 -q', 'col');

        %[g, acc, prob] = predict(test_label, test_inst, model, '', 'col');
        rank_label = rank_candidate(test_inst, model);
        acc = calculate_accuracy(test_label', rank_label);
toc

        top_acc(cv, :) = acc(1:min([10, length(acc)])) * 100;

        fprintf('\nTop-N accuracy\n');
        disp(top_acc(cv, :));
    end
    top_acc = [top_acc; mean(top_acc)]
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
        iter = 400;
        iter = round(size(vocabs, 2) / 512);
        fprintf('iter = %d\n', iter);

        param = struct('K', 1024, 'lambda', 0.25, 'lambda2', 0, ...
                       'iter', iter, 'mode', 2, ...
                       'modeD', 0, 'batchsize', 512, 'modeParam', 0, ...
                       'clean', true, 'numThreads', 4, 'verbose', false);
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
