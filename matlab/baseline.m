function feature_classify(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract descriptors if mat-file of dataset doesn't exist
    dataset_mat = [strrep(image_list, '.list', ''), '.mat'];
    if exist(dataset_mat, 'file') ~= 2
        norm_size = [256 256];
        dataset = extract_descriptors(dataset, norm_size);
        %save(dataset_mat, '-v7.3');
    else
        load(dataset_mat);
    end

    % For each fold, generate features by descriptors
    num_fold = 5;
    folds = cross_validation(dataset, num_fold);
    for v = 1:num_fold
        f = struct('train', folds(v).train, 'test', folds(v).test, 'v', v);
        label = [dataset.label];

        % SIFT descriptors with sparse coding
        sift = struct('dim', 1024, 'p', [], 'dict', [], 'alpha', [], 'n', []);
        sift.p = struct('K', sift.dim, 'lambda', 1, 'lambda2', 0, ...
                        'iter', 200/200, 'mode', 2, 'modeD', 0, ...
                        'modeParam', 0, 'clean', true, 'numThreads', 1);
        sift.dict = mexTrainDL_Memory([dataset(f.train).sift], sift.p);
        sift.alpha = mexLasso([dataset.sift], sift.dict, sift.p);
        sift.n = [dataset.sift_num];
        sift_encode = pooling(sift.alpha, sift.n);

        % LBP descriptors with sparse coding
        lbp = struct('dim', 2048/512, 'p', [], 'dict', [], 'alpha', [], 'n', []);
        lbp.p = struct('K', lbp.dim, 'lambda', 1, 'lambda2', 0, ...
                       'iter', 200/200, 'mode', 2, 'modeD', 0, ...
                       'modeParam', 0, 'clean', true, 'numThreads', 1);
        lbp.dict = mexTrainDL_Memory(double([dataset(f.train).lbp]), lbp.p);
        lbp.alpha = mexLasso(double([dataset.lbp]), lbp.dict, lbp.p);
        lbp.n = [dataset.lbp_num];
        lbp_encode = pooling(lbp.alpha, lbp.n);

        % Encode color histogram & Gabor filter bank response
        color_encode = [dataset.color];
        gabor_encode = [dataset.gabor];

        % Adaboost
        info = struct('label', label, 'train', f.train, 'test', f.test, 'v', 4);
        samme(info, sift_encode, lbp_encode, color_encode, gabor_encode);
        break

        % Write partial problem to disk
        %write_problem(['sift_', num2str(v)], label, sift_encode, f);
        %write_problem(['lbp_', num2str(v)], label, lbp_encode, f);
        %write_problem(['color_', num2str(v)], label, color_encode, f);
        %write_problem(['gabor_', num2str(v)], label, gabor_encode, f);

        % Learned by liblinear
        %c = 10.^[2:-1:-7];
        %sift_acc(v, :) = linear_classify(label, sift_encode, f, c);
        %lbp_acc(v, :) = linear_classify(label, lbp_encode, f, c);
        %color_acc(v, :) = linear_classify(label, color_encode, f, c);
        %gabor_acc(v, :) = linear_classify(label, gabor_encode, f, c);

        % Show current result and save temporary progress
        %[mean(sift_acc,1);mean(lbp_acc,1);mean(color_acc,1);mean(gabor_acc,1)]
        %save('warm_start.mat');
    end
end

function samme(varargin)
    label = varargin{1}.label;
    train_list = varargin{1}.train;
    test_list = varargin{1}.test;
    num_fold = varargin{1}.v;
    features = varargin(2:end);
    
    % Extract validation set for boosting
    categories = unique(label);
    val_list = [];
    for c = 1:length(categories)
        category_list = train_list(label(train_list)==categories(c));
        category_len = length(category_list);
        val_num = floor(category_len/num_fold);
        val_samples = category_list(randsample(category_len, val_num));
        val_list = [val_list val_samples];
        train_list = setdiff(train_list, val_samples);
    end

    for idx = 1:length(features)
    end
end
