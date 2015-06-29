function baseline(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    fprintf('%s\n%s\n', repmat(['-'], 1, 80), datestr(datetime('now')));
    tic
        % Extract descriptors if mat-file of dataset doesn't exist
        dataset_name = strrep(image_list, '.list', '');
        dataset_mat = [dataset_name, '.mat'];
        if exist(dataset_mat, 'file') ~= 2
            norm_size = [256 256];
            dataset = extract_descriptors(dataset, norm_size);
            %save([dataset_name, '.mat'], '-v7.3');
        else
            %load([dataset_name, '.mat']);
        end
    toc
    fprintf('\n');

    % For each fold, generate features by descriptors
    num_fold = 5;
    folds = cross_validation(dataset, num_fold);
    for v = 1:num_fold
        f = struct('train', folds(v).train, 'test', folds(v).test, 'val', []);
        label = double([dataset.label]');
        encode = struct('sift', [], 'lbp', [], 'color', [], 'gabor', []);
        encode_name = fieldnames(encode);
        num_inst = length(label);

        % SIFT descriptors with sparse coding
        tic
            batchsize = 16384;
            iter = ceil(sum([dataset.sift_num]) / batchsize);
            sift = struct('dim', 1024, 'p', [], ...
                          'dict', [], 'alpha', [], 'n', []);
            sift.p = struct('K', sift.dim, 'lambda', 0.25, 'lambda2', 0, ...
                            'iter', iter, 'batchsize', batchsize, ...
                            'mode', 2, 'modeD', 0, 'modeParam', 0, ...
                            'clean', true, 'numThreads', 4, 'verbose', false);
            sift.dict = mexTrainDL([dataset(f.train).sift], sift.p);
        toc
            sift.alpha = mexLasso([dataset.sift], sift.dict, sift.p);
            sift.n = [dataset.sift_num];
            encode.sift = pooling(sift.alpha, sift.n)';
            encode.sift = sparse(scale_data(encode.sift, f));
        toc


        % LBP descriptors with sparse coding
            batchsize = 16384;
            iter = ceil(sum([dataset.lbp_num]) / batchsize);
            lbp = struct('dim', 2048, 'p', [], ...
                         'dict', [], 'alpha', [], 'n', []);
            lbp.p = struct('K', lbp.dim, 'lambda', 0.25, 'lambda2', 0, ...
                           'iter', iter, 'batchsize', batchsize, 'mode', 2, ...
                           'modeD', 0, 'modeParam', 0, 'clean', true, ...
                           'numThreads', 4, 'verbose', false);
            lbp.dict = mexTrainDL([dataset(f.train).lbp], lbp.p);
        toc
            lbp.alpha = mexLasso([dataset.lbp], lbp.dict, lbp.p);
            lbp.n = [dataset.lbp_num];
            encode.lbp = pooling(lbp.alpha, lbp.n)';
            encode.lbp = sparse(scale_data(encode.lbp, f));
        toc


        % Color histogram & Gabor filter bank response
            encode.color = sparse([dataset.color]');
            encode.color = scale_data(encode.color, f);
        toc
            encode.gabor = sparse([dataset.gabor]');
            encode.gabor = scale_data(encode.gabor, f);
        toc

        %% Write subproblem for grid.py & warm start
        %for idx = 1:numel(encode_name)
        %    name = encode_name{idx};
        %    trainfile = [dataset_name, '_', name, '_', num2str(v), '.train'];
        %    testfile = [dataset_name, '_', name, '_', num2str(v), '.test'];
        %    libsvmwrite(trainfile, label(f.train), encode.(name)(f.train, :));
        %    libsvmwrite(testfile, label(f.test), encode.(name)(f.test, :));
        %end

        % Extract validation set for linear blending on base learner
        num_fold_val = 4;
        f.val = extract_val_list(label, f.train, num_fold_val);
        f.train = setdiff(f.train, f.val);

        % Learn RBF-SVM classifier as base learner
        fprintf('\nBase learner\n');
        tic
            option = struct('sift', '-c 8 -b 1 -q', ...
                            'lbp', '-c 8 -b 1 -q', ...
                            'color', '-c 8 -b 1 -q', ...
                            'gabor', '-c 8 -b 1 -q');
            train_label = label(f.train);
            for idx = 1:numel(encode_name)
                name = encode_name{idx};
                train_inst = encode.(name)(f.train, :);
                base.(name) = svmtrain(train_label, train_inst, option.(name));
            end
        toc

        % Linear blending by multi-class Adaboost with SAMME
        fprintf('SAMME\n')
            t_max = 10000;
            ballot = linear_blend(t_max, base, label, encode, f);
            ballot_list(v, :) = ballot;
        toc

        % Testing with weighted ballot & probability estimation by libsvm
        fprintf('\nPredict\n');
        tic
            prob_est = zeros(numel(f.test), length(unique(label)));
            test_label = label(f.test);
            for base_idx = 1:numel(encode_name)
                name = encode_name{base_idx};
                test_inst = encode.(name)(f.test, :);
                model = base.(name);
                [g, ~, p] = svmpredict(test_label, test_inst, model, '-b 1');
                prob_est = prob_est + ballot.(name)*p;
            end
        toc

        num_test = numel(f.test);
        [~, rank] = sort(prob_est, 2, 'descend');
        for rank_idx = 1:size(rank, 2)
            is_correct = (model.Label(rank(:, rank_idx)) == test_label);
            acc(rank_idx) = sum(is_correct) / num_test;
        end
        top_acc(v, :) = cumsum(acc);

        top_n = min(10, size(top_acc, 2));
        fprintf('\nTop-1 accuracy: %.2f%%\n\n\n\n', top_acc(v, 1) * 100);
    end

    fprintf('Top-%d Accuracy: \n\n', top_n);
    disp([top_acc(:, 1:top_n); mean(top_acc(:, 1:top_n), 1)])
end


function data_hat = scale_data(data, fold)

    num_inst = size(data, 1);
    data_mean = repmat(mean(data(fold.train, :)), num_inst, 1);
    data_std = repmat(std(data(fold.train, :)), num_inst, 1);
    find(data_std == 0) = 1;

    data_hat = (data - data_mean) ./ data_std;
end
