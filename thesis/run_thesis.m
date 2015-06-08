function run_thesis(image_list)

    % -------------------------------------------------------------------------
    % Parse image file list
    % -------------------------------------------------------------------------

    setup_3rdparty(fullfile('~/Software'))
    [prefix, label, path] = parse_list(image_list);
    folds = cross_validation(label, 5);

    % -------------------------------------------------------------------------
    % Feature Extraction
    % -------------------------------------------------------------------------

    %sift = extract_descriptor(path, 'sift');
    %lbp = extract_descriptor(path, 'lbp');
    %hog = extract_descriptor(path, 'hog');
    phow = extract_descriptor(path, 'phow');

    % -------------------------------------------------------------------------
    % Directly Training
    % -------------------------------------------------------------------------

    %lbp_feature = [lbp{:, :}];
    %lbp_feature = cell2mat(reshape(lbp_feature, 1, []));
    %lbp_feature = reshape(lbp_feature, [], size(lbp, 2));
    %train(double(label), sparse(double(lbp_feature)), '-v 5 -q', 'col');

    %hog_feature = reshape(hog, 1, 1, []);
    %hog_feature = reshape(cell2mat(hog_feature), [], size(hog, 2));
    %hog_feature = normc(double(hog_feature));
    %train(double(label), sparse(double(hog_feature)), '-v 5 -q', 'col');

    %phow_feature = reshape(cell2mat(phow), [], size(phow, 2));
    %phow_feature = normc(double(phow_feature));
    %train(double(label), sparse(double(phow_feature)), '-v 5 -q', 'col');

    %save([prefix, '.mat'], 'sift', 'lbp')

    % -------------------------------------------------------------------------
    % Bag-of-Word with Hierarchical K-means Codebook, Encoding with LLC
    % -------------------------------------------------------------------------

    for cv = 1:length(folds) / length(folds)
        train_idx = folds(cv).train;
        test_idx = folds(cv).test;

        % Generate bag-of-word histogram with codebook
        branch = 2;
        level = 10;
        dict = kmeans_codebook(cell2mat(phow(train_idx)), branch, level);
        dict = llc_codebook(cell2mat(phow(train_idx)), branch ^ level);

        %encode = bag_of_word(dict, sift);
        encode = llc_encode(dict, phow);

        % Approximated chi-square kernel mapping
        %encode = vl_homkermap(ds_encode, 2, 'kernel', 'kchi2');

        % Evaluate with linear svm
        train_inst = sparse(encode(:, train_idx));
        test_inst = sparse(encode(:, test_idx));
        model = train(double(label(train_idx)), train_inst, '-q', 'col');
        predict(double(label(test_idx)), test_inst, model, '', 'col');
    end
end

function setup_3rdparty(root_dir)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile(root_dir, 'vlfeat/toolbox/vl_setup'));
    addpath(fullfile(root_dir, 'liblinear/matlab'));
    addpath(fullfile(root_dir, 'libsvm/matlab'));

    % Add SPAMS(SPArse Modeling Software) path
    addpath(fullfile(root_dir, 'spams-matlab/build'));
    addpath(fullfile(root_dir, 'spams-matlab/test_release'));
    addpath(fullfile(root_dir, 'spams-matlab/src_release'));
end

function [prefix, label, path] = parse_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    fclose(fd);

    [path, label] = raw{:};
    [~, prefix, ~] = fileparts(image_list);
end

function norm_image = normalize_image(image, norm_size, crop)
    if nargin < 3, crop=true; end

    if not(crop)
        norm_image = imresize(image, norm_size);
    else
        [height, width, channel] = size(image);
        scale = max(norm_size./[height, width])+eps;
        offset = floor(([height width]*scale - norm_size)/2);
        x = offset(2)+1:offset(2)+norm_size(2);
        y = offset(1)+1:offset(1)+norm_size(1);
        resized_image = imresize(image, scale);
        norm_image = resized_image(y, x, :);
    end
end

function image = read_image(path)
    norm_size = [256, 256];

    % TODO: Check if image is valid 3-channel image
    raw_image = imread(path);
    image = normalize_image(raw_image, norm_size);
end

function folds = cross_validation(label, num_fold)
    folds(1:num_fold) = struct('train', [], 'test', []);
    categories = unique(label);
    for c = 1:length(categories)
        % Select particular category
        list = find(label == categories(c));
        len = length(list);

        % Calculate #test_case on each fold, exactly cover all instances
        sample_fold = randsample(num_fold, mod(len, num_fold));
        test_nums(1:num_fold) = floor(len/num_fold);
        test_nums(sample_fold) = floor(len/num_fold)+1;
        test_nums = test_nums - (test_nums==len);  % Ensure #train_instance > 0

        % Distribute all instances to training set and testing set
        list = list(randperm(len));
        for v = 1:num_fold
            test_list = list(1:test_nums(v));
            train_list = list(test_nums(v)+1:end);

            folds(v).train = [folds(v).train; train_list];
            folds(v).test = [folds(v).test; test_list];
            list = [train_list; test_list];
        end
    end
end

function descriptor = extract_descriptor(path, ds_type)
    descriptors = cell(1, length(path));
    for idx = 1:length(path)
        image = read_image(path{idx});
        gray_image = rgb2gray(image);

        switch ds_type
            case 'sift'
                descriptor{idx} = get_sift(single(gray_image));
            case 'lbp'
                descriptor{idx} = get_pyramid_lbp(single(gray_image));
            case 'hog'
                descriptor{idx} = get_hog(single(image));
            case 'phow'
                descriptor{idx} = get_phow(im2single(image));
            otherwise
                fprintf(1, 'Wrong descriptor name.');
                return;
        end
    end
end

function ds = get_sift(image)
    [fs, ds] = vl_sift(image);
end

function ds = get_pyramid_lbp(image)
    level = 3;
    scale = 1 / 2;

    ds = cell(level, 1);
    blur_kernel = fspecial('gaussian', [9 9], 1.6);

    ds{1} = get_lbp(image);
    for lv = 2:level
        image = imfilter(image, blur_kernel, 'symmetric');
        image = imresize(image, scale);
        ds{lv} = get_lbp(image);
    end
end

function lbp = get_lbp(image)
    step_size = 32;
    window_size = 64;

    lbp_window = vl_lbp(image, window_size);
    lbp_overlap = vl_lbp(image(step_size:end, step_size:end), window_size);

    lbp = [reshape(lbp_window, [], 58)', reshape(lbp_overlap, [], 58)'];
end

function ds = get_hog(image)
    cell_size = 16;
    ds = vl_hog(image, cell_size, 'numOrientations', 64);
end

function ds = get_phow(image)
    [fs, ds] = vl_phow(image, 'Color', 'gray', 'Sizes', [8 12 16], ...
                       'Step', 8, 'WindowSize', 2, 'Magnif', 6);
end

function codebook = kmeans_codebook(vocab, branch, level)
    leaves = branch ^ level;

    tree = vl_hikmeans(vocab, branch, leaves, 'Method', 'lloyd', 'MaxIters', 400);
    codebook = get_leaves_center(tree);
end

function centers = get_leaves_center(tree)
    if tree.depth == 1
        centers = tree.centers;
    else
        centers = [];
        queue = tree.sub;
        while ~isempty(queue)
            if isempty(queue(1).sub)
                centers = [centers queue(1).centers];
            else
                queue = [queue queue(1).sub];
            end
            queue(1) = [];
        end
    end
end

function bow = bag_of_word(dict, vocabs)
    dict_size = size(dict, 2);

    bow = zeros(dict_size, length(vocabs));
    for idx = 1:length(vocabs)
        asgn = vl_ikmeanspush(vocabs{idx}, dict);
        hist = vl_ikmeanshist(dict_size, asgn);
        bow(:, idx) = double(hist) / sum(hist);
    end
end

function llc = llc_encode(dict, vocabs)
    dict = double(dict) / 255.0;
    llc = zeros(size(dict, 2), length(vocabs));
    for idx = 1:length(vocabs)
        x = double(vocabs{idx}) / 255.0;

        % Exactly solution of LLC
        %sigma = 1.0;
        %lambda = 1.0;
        %llc_coeff = llc_exact(dict, x, sigma, lambda);

        % Approximate solution of LLC
        knn = 5;
        llc_coeff = llc_approx(dict, x, knn);

        % Pooling
        llc(:, idx) = mean(llc_coeff, 2);
        %llc(:, idx) = max(llc_coeff, [], 2);

        % Normalization
        llc(:, idx) = llc(:, idx) / norm(llc(:, idx));
        %llc(:, idx) = llc(:, idx) / sum(llc(:, idx));
    end
end

function llc = llc_exact(B, X, sigma, lambda)
    x_num = size(X, 2);
    b_num = size(B, 2);

    llc = zeros(b_num, x_num);
    for idx = 1:size(X, 2)
        xi = X(:, idx);
        di = exp(vl_alldist2(B, xi, 'L2') / sigma);
        di = di / max(di);

        z = B - repmat(xi, 1, b_num);                % Shift properties
        Ci = z' * z;                                 % Local covariance
        Ci = Ci + eye(b_num) * trace(Ci) * 1e-4;     % Regularization
        ci = (Ci + lambda * diag(di .* di)) \ ones(b_num, 1); 
        llc(:, idx) = ci / sum(ci);
    end
end

function llc = llc_approx(B, X, knn)
    x_num = size(X, 2);
    b_num = size(B, 2);

    kd_tree = vl_kdtreebuild(B);
    nearest_neighbors = vl_kdtreequery(kd_tree, B, X, 'NumNeighbors', knn);
    llc = zeros(b_num, x_num);
    for idx = 1:size(X, 2)
        xi = X(:, idx);
        nn = nearest_neighbors(:, idx);

        z = B(:, nn) - repmat(xi, 1, knn);   % Shift properties
        C = z' * z;                          % Local covariance
        C = C + eye(knn) * trace(C) * 1e-4;  % Regularization
        w = C \ ones(knn, 1);

        llc(nn, idx) = w / sum(w);
    end
end
