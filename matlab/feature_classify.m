function feature_classify(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract descriptors
    dataset = extract_descriptors(dataset);
    %save('baseline.mat', '-v7.3');

    % For each fold, generate features by descriptors
    num_fold = 5;
    folds = cross_validation(dataset, num_fold);
    for v = 1:num_fold/num_fold
        train_set = dataset(folds(v).train);
        test_set = dataset(folds(v).test);
        sc = @(x, d, p) mean(mexLasso(double(x), d, p), 2);

        % Encode SIFT descriptor
        param = struct('K', 1024/512, 'lambda', 1, 'lambda2', 0, 'iter', 1000/1000, 'numThreads', -1);
        sift_dict = mexTrainDL_Memory(double([train_set.sift]), param);
        sift = arrayfun(@(x) {sc(x.sift, sift_dict, param)}, dataset');

        % Encode LBP descriptor
        param = struct('K', 2048/1024, 'lambda', 1, 'lambda2', 0, 'iter', 1000/1000, 'numThreads', -1);
        lbp_dict = mexTrainDL_Memory(double([train_set.lbp]), param);
        lbp = arrayfun(@(x) {sc(x.lbp, lbp_dict, param)}, dataset);
        
        % Encode color histogram
        color = [dataset.color];

        % Encode Gabor filter bank response
        gabor = [dataset.gabor];

        % Classify by linear SVM
        c = 10.^[2:-1:-7];
%{
        model = train(labels.train, sparse(features.train), ...
                      ['-c ', num2str(c(idx)), ' -q'], 'col');
        [~, acc, ~] = predict(labels.test, ...
                              sparse(features.test), model, '-q', 'col');
        acc_list(idx) = acc(1);
%}
 

%{
        sift_acc(v, :) = linear_classify(sift, labels, c);
        lbp_acc(v, :) = linear_classify(lbp, labels, c);
        color_acc(v, :) = linear_classify(color, labels, c);
        gabor_acc(v, :) = linear_classify(gabor, labels, c);
%}
    end
%{
    %save('baseline.mat', '-v7.3');
    sift_acc = [c; mean(sift_acc)]
    lbp_acc = [c; mean(lbp_acc)]
    color_acc = [c; mean(color_acc)]
    gabor_acc = [c; mean(gabor_acc)]
%}
end

