function feature_classify(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract descriptors
    norm_size = [256 256];
    dataset = extract_descriptors(dataset, norm_size);
    save(strrep(image_list, '.list', '.mat'), '-v7.3');

    % For each fold, generate features by descriptors
    num_fold = 5;
    folds = cross_validation(dataset, num_fold);
    for v = 1:num_fold
        train_set = folds(v).train;
        test_set = folds(v).test;

        % Generate sparse coding dictionary of SIFT & LBP
        sift_dim = 1024;
        lbp_dim = 2048;
        ps = struct('K', sift_dim, 'lambda', 1, 'lambda2', 0, 'mode', 2, ...
                        'iter', 500, 'verbose', false, 'numThreads', 17);
        pl = struct('K', lbp_dim, 'lambda', 1, 'lambda2', 0, 'mode', 2, ...
                    'iter', 500, 'verbose', false, 'numThreads', 17);
        [sift_dict, ms] = mexTrainDL(double([dataset(train_set).sift]), ps);
        [lbp_dict, ml] = mexTrainDL(double([dataset(train_set).lbp]), pl);

        % Encode SIFT & LBP with sparse coding
        sift_alpha = mexLasso(double([dataset.sift]), sift_dict, ps);
        lbp_alpha = mexLasso(double([dataset.lbp]), lbp_dict, pl);
        sift_num = [dataset.sift_num];
        lbp_num = [dataset.lbp_num];
        sift = zeros(sift_dim, length(dataset));
        lbp = zeros(lbp_dim, length(dataset));
        for idx = 1:length(dataset)
            sift(:, idx) = mean(sift_alpha(:, 1:sift_num(idx)), 2);
            lbp(:, idx) = mean(lbp_alpha(:, 1:lbp_num(idx)), 2);
            sift_alpha = sift_alpha(:, sift_num(idx)+1:end);
            lbp_alpha = lbp_alpha(:, lbp_num(idx)+1:end);
        end

        % Encode color histogram & Gabor filter bank response
        color = [dataset.color];
        gabor = [dataset.gabor];

        % Classify by linear SVM
        sift_acc(v, :) = linear_classify([dataset.label], [sift], folds(v));
        lbp_acc(v, :) = linear_classify([dataset.label], [lbp], folds(v));
        color_acc(v, :) = linear_classify([dataset.label], color, folds(v));
        gabor_acc(v, :) = linear_classify([dataset.label], gabor, folds(v));
        save('warm_start', '-v7.3');
    end
    sift_acc = [mean(sift_acc)]
    lbp_acc = [mean(lbp_acc)]
    color_acc = [mean(color_acc)]
    gabor_acc = [mean(gabor_acc)]
end

