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

        dataset(idx).sift = extract_sift(norm_image);
        dataset(idx).lbp = extract_lbp(norm_image);
        dataset(idx).color = extract_color(norm_image);
        dataset(idx).gabor = extract_gabor(norm_image);
    end
    save('descriptor.mat', '-v7.3');

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
        sift_size = 1024;
        lbp_size = 2048;
        %sift = kmeans_encode([train_list.sift]', [test_list.sift]', sift_size);
        %lbp = kmeans_encode([train_list.lbp]', [test_list.lbp]', lbp_size);
        sift = sparse_encode([train_list.sift]', [test_list.sift]', sift_size);
        lbp = sparse_encode([train_list.lbp]', [test_list.lbp]', lbp_size);
        color = struct('train', [train_list.color], 'test', [test_list.color]);
        gabor = struct('train', [train_list.gabor], 'test', [test_list.gabor]);

        % Classify by linear SVM
        c = 10.^[2:-1:-7];
        sift_acc(v, :) = linear_classify(sift, labels, c);
        lbp_acc(v, :) = linear_classify(lbp, labels, c);
        color_acc(v, :) = linear_classify(color, labels, c);
        gabor_acc(v, :) = linear_classify(gabor, labels, c);
    end
    save('descriptor.mat', '-v7.3');
    sift_acc = [c; mean(sift_acc)]
    lbp_acc = [c; mean(lbp_acc)]
    color_acc = [c; mean(color_acc)]
    gabor_acc = [c; mean(gabor_acc)]
end
