function run_thesis(image_list)
    [~, prefix, ~] = fileparts(image_list);
    [path, label, food] = parse_list(image_list);
    folds = cross_validation(label, 5);

    fprintf('-------------------------------------------------------------\n');
    tic
        lbp = extract_lbp(path);
    toc; 

    for cv = 1:length(folds)
        train_idx = folds(cv).train;
        test_idx = folds(cv).test;

        % Augmented data by flip LBP
        ds = [lbp{1, :}, lbp{2, train_idx}];
        aug_label = [label; label(train_idx)];
        aug_train_idx = setdiff(1:size(ds, 2), test_idx)';
        %aug_train_idx = train_idx;     % Delete augmented training data

        tic
            % Generate descriptor basis
            basis = generate_basis(ds(:, aug_train_idx));
        toc

        tic
            % Encode descriptors and do approximate kernel mapping
            feature = encode_descriptor(basis, ds);
            feature = cell2mat(feature);
            feature = vl_homkermap(feature, 3, 'kernel', 'kinters', 'gamma', 1);
        toc

        tic
            % Linear classification
            train_inst = sparse(feature(:, aug_train_idx));
            train_label = double(aug_label(aug_train_idx));
            test_inst = sparse(feature(:, test_idx));
            test_label = double(aug_label(test_idx));

            % Generate model & predict cantidate
            model = train(train_label, train_inst, '-s 1 -c 10 -q', 'col');
            rank_label = rank_candidate(test_inst, model);
            %[g, acc, v] = predict(test_label, test_inst, model, '', 'col');
        toc

        % Store classification report
        report(cv) = classification_report(test_label', rank_label);
        acc = report(cv).accuracy(1);
        fprintf('Top-1 Accuracy: %.2f%%\n\n', acc * 100);
    end

    % Save report for furthur research
    %save([prefix, '_result.mat'], 'report');

    % Confusion matrix
    cm = sum(cat(3, report(:).confusion_matrix), 3)

    % Average precision & recall
    var_name = {'Precision', 'Recall'};
    pr = cat(1, report(:).precision);
    rc = cat(1, report(:).recall);
    metric = [mean(pr); mean(rc)]';
    metric = array2table(metric, 'VariableNames', var_name, 'RowNames', food)

    % Top-N accuracy
    top_n = min(size(acc, 2), 10);
    acc = cat(1, report(:).accuracy);
    top_acc = [acc(:, 1:top_n); mean(acc(:, 1:top_n))]
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
        batch_size = 16384;
        iter = round(size(vocabs, 2) / batch_size);
        param = struct('K', 1024, 'lambda', 0.25, 'lambda2', 0, ...
                       'iter', iter, 'mode', 2, 'modeD', 0, ...
                       'batchsize', batch_size, 'modeParam', 0, ...
                       'clean', true, 'numThreads', 4, 'verbose', false);
        fprintf('K = %4d, iter = %4d, batch = %5d\n', ...
                param.K, param.iter, param.batchsize);
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
