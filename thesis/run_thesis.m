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

    sift = extract_descriptor(path, 'sift');
    %lbp = extract_descriptor(path, 'lbp');
    %hog = extract_descriptor(path, 'hog');
    %covdet = extract_descriptor(path, 'covdet');

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
    % Bag-of-Word with Hierarchical K-means Codebook
    % -------------------------------------------------------------------------

    train_idx = folds(1).train;
    test_idx = folds(1).test;

    % Generate bag-of-word histogram with codebook
    branch = 4;
    level = 4;
    sift_tree = generate_codebook(cell2mat(sift(train_idx)), branch, level);
    sift_bow = kmeans_bag_of_word(sift_tree, sift);

    % Approximated chi-square kernel mapping
    sift_bow = vl_homkermap(sift_bow, 2, 'kernel', 'kchi2');

    % Evaluate with linear svm
    train_sift = sparse(sift_bow(:, train_idx));
    test_sift = sparse(sift_bow(:, test_idx));
    model = train(double(label(train_idx)), train_sift, '-q', 'col');
    predict(double(label(test_idx)), test_sift, model, '', 'col');
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
            case 'covdet'
                descriptor{idx} = get_covdet(im2single(image));
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

function ds = get_covdet(image)
    %fs = vl_covdet
    [fs, ds] = vl_phow(image, 'Color', 'gray', 'Sizes', [16 32], 'Step', 64);
end

function tree = generate_codebook(vocab, branch, level)
    leaves = branch ^ level;
    tree = vl_hikmeans(vocab, branch, leaves, 'method', 'elkan', 'verbose');
end

function hbow = kmeans_bag_of_word(tree, descriptor)
    leaves = tree.K ^ tree.depth;

    hbow = zeros(leaves, length(descriptor));
    for idx = 1:length(descriptor)
        path = vl_hikmeanspush(tree, descriptor{idx});
        hist = vl_hikmeanshist(tree, path);

        leave_hist = hist(end - leaves + 1:end);
        hbow(:, idx) = double(leave_hist) / sum(leave_hist);
    end
end
