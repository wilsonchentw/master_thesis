function feature_classify(image_list)
    setup_3rdparty();

    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract image descriptors
    norm_size = [64 64];
    for idx = 1:length(dataset)
        % Read and preprocessing image
        image = imread(dataset(idx).path);
        norm_image = normalize_image(image, norm_size, true);

        % Extract SIFT descriptors
        dataset(idx).sift = extract_sift(norm_image);

        % Extract LBP descriptors
        dataset(idx).lbp = extract_lbp(norm_image);
    end

    % For each fold, generate features by descriptors
    num_fold = 5;
    folds = cross_validation(dataset, num_fold);
    for v = 1:num_fold
        train_list = dataset(folds(v).train);
        test_list = dataset(folds(v).test);

        labels = struct('train', [], 'test', []);
        labels.train = double([train_list.label]');
        labels.test = double([test_list.label]');

        % Generate features by descriptors
        dict_size = 1024/512;
        %sift = kmeans_encode([train_list.sift]', [test_list.sift]', dict_size);
        %lbp = kmeans_encode([train_list.lbp]', [test_list.lbp]', dict_size);
        sift = sparse_encode([train_list.sift]', [test_list.sift]', dict_size);

        % Classify by linear SVM
        c = 10.^[2:-1:-7];
        sift_acc(v, :) = linear_classify(sift, labels, c);
        %lbp_acc(v, :) = linear_classify(lbp, labels, c);
    end

    sift_acc = [c; mean(sift_acc)]
    %lbp_acc = [c; mean(lbp_acc)]
end

function setup_3rdparty()
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Add SPAMS(SPArse Modeling Software) path
    addpath('../spams/spams-matlab/test_release');
    addpath('../spams/spams-matlab/src_release');
    addpath('../spams/spams-matlab/build');
end

function features = sparse_encode(train_list, test_list, dict_size)
    features = struct('train', zeros(dict_size, length(train_list)), ...
                      'test', zeros(dict_size, length(test_list)));
    param = struct('K', dict_size, 'lambda', 1, 'iter', 100);

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

