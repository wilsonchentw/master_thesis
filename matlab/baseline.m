function baseline(image_list)
    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract descriptors if mat-file of dataset doesn't exist
    dataset_mat = [strrep(image_list, '.list', ''), '.mat'];
    if exist(dataset_mat, 'file') ~= 2
        norm_size = [256 256];
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
        sift = struct('dim', 1024/512, 'p', [], 'dict', [], 'alpha', [], 'n', []);
        sift.p = struct('K', sift.dim, 'lambda', 1, 'lambda2', 0, ...
                        'iter', 1000/1000, 'mode', 2, 'modeD', 0, ...
                        'modeParam', 0, 'clean', true, 'numThreads', 4);
        sift.dict = mexTrainDL_Memory([dataset(f.train).sift], sift.p);
        sift.alpha = mexLasso([dataset.sift], sift.dict, sift.p);
        sift.n = [dataset.sift_num];
        sift_encode = pooling(sift.alpha, sift.n);

        % LBP descriptors with sparse coding
        lbp = struct('dim', 2048/512, 'p', [], 'dict', [], 'alpha', [], 'n', []);
        lbp.p = struct('K', lbp.dim, 'lambda', 0.1, 'lambda2', 0, ...
                       'iter', 1000/1000, 'mode', 2, 'modeD', 0, ...
                       'modeParam', 0, 'clean', true, 'numThreads', 4);
        lbp.dict = mexTrainDL_Memory([dataset(f.train).lbp], lbp.p);
        lbp.alpha = mexLasso([dataset.lbp], lbp.dict, lbp.p);
        lbp.n = [dataset.lbp_num];
        lbp_encode = pooling(lbp.alpha, lbp.n);

        % Encode color histogram & Gabor filter bank response
        color_encode = [dataset.color];
        gabor_encode = [dataset.gabor];

        % Multi-class Adaboost by SAMME
        inst = {sift_encode, lbp_encode, color_encode, gabor_encode};
        samme(label, inst, f)
        break
    end
end

function samme(label, inst, fold)
    label = double(label);
    train_list = fold.train;
    test_list = fold.test;

    % Write subproblem for grid.py to search best parameter
    %for idx = 1:length(inst)
    %    filename = ['feature_', num2str(idx), '.train'];
    %    train_inst = sparse(inst{idx}(:, train_list)')
    %    libsvmwrite(filename, label(train_list), train_inst);
    %end
    train_option = {'-c 1 -g 0.0010 -b 1 -q', '-c 1 -g 0.0005 -b 1 -q', ...
                    '-c 1 -g 0.0007 -b 1 -q', '-c 1 -g 0.0010 -b 1 -q', };
    val_option = '-b 0';
    test_option = '-b 1 -q';

    % Extract validation set for boosting
    num_fold = 4;
    val_list = extract_val_list(label, train_list, num_fold);
    train_list = setdiff(train_list, val_list);

    % Learn base classifier by libsvm
    for idx = 1:length(inst)
        train_inst = sparse(inst{idx}(:, train_list)');
        train_label = label(train_list)';
        val_inst = sparse(inst{idx}(:, val_list)');
        val_label = label(val_list)';

        base(idx) = svmtrain(train_label, train_inst, train_option{idx});
        [g, acc, p] = svmpredict(val_label, val_inst, base(idx), val_option);
        prob_est(:, :, idx) = p;
        guess_label(:, idx) = g;
        is_correct(:, idx) = (g == val_label);
    end

    % Generate fake probability by predict label (FOR DEBUG PURPOSE)
    prob_est = zeros(length(val_label), length(unique(label)), length(inst));
    for base_idx = 1:length(base)
        for inst_idx = 1:length(val_label)
            prob_est(inst_idx, guess_label(inst_idx, base_idx), base_idx) = 1;
        end
    end

    % Generate linear blending coefficient (alpha) by SAMME
    t_max = 5000/5000;
    num_category = length(unique(label));
    num_val = length(val_label);
    w = ones(num_val, 1)/num_val;
    vote = zeros(length(base), 1);
    for t = 1:t_max
        
    end

%{
    % Generate linear blending coefficient (alpha) by SAMME
    t_max = 5000;
    vote = zeros(1, length(inst));
    num_category = length(unique(label));
    w = ones(length(val_label), 1)/length(val_label);
    for t = 1:t_max
        % Select weak learner
        score = w'*is_correct;
        [~, weak] = max(vote*score);

        % Update
        err = w'*(is_correct(:, weak)~=true)/sum(w);
        if err >= (num_category-1)/num_category
            break;
        end
        alpha = log((1-err)/err)+log(num_category-1);
        w = w.*exp(alpha*(is_correct(:, weak)~=true));
        w = w/sum(w);

        vote(weak) = vote(weak)+alpha;
    end
%}
end

