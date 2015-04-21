function feature_classify(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract descriptors
    norm_size = [256 256];
    dataset = extract_descriptors(dataset, norm_size);
    save('baseline.mat', '-v7.3');

    % For each fold, generate features by descriptors
    num_fold = 5;
    folds = cross_validation(dataset, num_fold);
    for v = 1:num_fold
        train_set = folds(v).train;
        test_set = folds(v).test;
        sc = @(x, d, p) mean(mexLasso(double(x), d, p), 2);

        % Encode SIFT descriptor
        param = struct('K', 1024, 'lambda', 1, 'lambda2', 0, ...
                       'iter', 1000, 'verbose', false, 'numThreads', 17);
        sift_dict = mexTrainDL_Memory(double([dataset(train_set).sift]), param);
        sift = arrayfun(@(x) {sc(x.sift, sift_dict, param)}, dataset');

        % Encode LBP descriptor
        param = struct('K', 2048, 'lambda', 1, 'lambda2', 0, ...
                       'iter', 1000, 'verbose', false, 'numThreads', 17);
        lbp_dict = mexTrainDL_Memory(double([dataset(train_set).lbp]), param);
        lbp = arrayfun(@(x) {sc(x.lbp, lbp_dict, param)}, dataset);
        
        % Encode color histogram
        color = [dataset.color];

        % Encode Gabor filter bank response
        gabor = [dataset.gabor];

        % Classify by linear SVM
        sift_acc(v, :) = linear_classify([dataset.label], [sift{:}], folds(v));
        lbp_acc(v, :) = linear_classify([dataset.label], [lbp{:}], folds(v));
        color_acc(v, :) = linear_classify([dataset.label], color, folds(v));
        gabor_acc(v, :) = linear_classify([dataset.label], gabor, folds(v));
    end
    save('baseline.mat', '-v7.3');
    sift_acc = [mean(sift_acc)]
    lbp_acc = [mean(lbp_acc)]
    color_acc = [mean(color_acc)]
    gabor_acc = [mean(gabor_acc)]
end
