function feature_classify(image_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract image descriptors
    norm_size = [64 64];
    dataset(length(dataset)).sift = [];
    dataset(length(dataset)).lbp = [];
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
        sift = kmeans_encode([train_list.sift]', [test_list.sift]', dict_size);
        lbp = kmeans_encode([train_list.lbp]', [test_list.lbp]', dict_size);

        % Classify by linear SVM
        c = 10.^[2:-1:-7];
        sift_acc(v, :) = linear_classify(sift, labels, c);
        lbp_acc(v, :) = linear_classify(lbp, labels, c);
    end

    sift_acc = [c; mean(sift_acc)]
    lbp_acc = [c; mean(lbp_acc)]
    %save('sift.mat');
end


