function feature_classify(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract descriptors if mat-file of dataset doesn't exist
    norm_size = [256 256];
    dataset_mat = [strrep(image_list, '.list', ''), '.mat'];
    if exist(dataset_mat, 'file') ~= 2
        dataset = extract_descriptors(dataset, norm_size);
        save(dataset_mat, '-v7.3');
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
                        'iter', 200, 'numThreads', 17);
        sift.dict = mexTrainDL_Memory(double([dataset(f.train).sift]), sift.p);
        sift.alpha = mexLasso(double([dataset.sift]), sift.dict, sift.p);
        sift.n = [dataset.sift_num];
        sift_encode = pooling(sift.alpha, sift.n);

        % LBP descriptors with sparse coding
        lbp = struct('dim', 2048, 'p', [], 'dict', [], 'alpha', [], 'n', []);
        lbp.p = struct('K', lbp.dim, 'lambda', 1, 'lambda2', 0, ...
                       'iter', 200, 'numThreads', 17);
        lbp.dict = mexTrainDL_Memory(double([dataset(f.train).lbp]), lbp.p);
        lbp.alpha = mexLasso(double([dataset.lbp]), lbp.dict, lbp.p);
        lbp.n = [dataset.lbp_num];
        lbp_encode = pooling(lbp.alpha, lbp.n);

        % Encode color histogram & Gabor filter bank response
        color_encode = [dataset.color];
        gabor_encode = [dataset.gabor];

        % Write problem for furthur usage
        write_problem(['sift_', num2str(v)], label, sift_encode, f);
        write_problem(['lbp_', num2str(v)], label, lbp_encode, f);
        write_problem(['color_', num2str(v)], label, color_encode, f);
        write_problem(['gabor_', num2str(v)], label, gabor_encode, f);

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
