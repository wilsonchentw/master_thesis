function feature_classify(image_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Parse image list into structure array    
    dataset = parse_image_list(image_list);

    % Extract image descriptors
    norm_size = [64 64];
    dataset(length(dataset)).sift = struct('sift', []);
    for idx = 1:length(dataset)
        % Read and preprocessing image
        image = imread(dataset(idx).path);
        norm_image = normalize_image(image, norm_size, true);

        % Extract SIFT descriptors
        dataset(idx).sift = extract_sift(image);
    end

    % For each fold, generate features by descriptors
    num_fold = 5;
    datasets = cross_validation(dataset, num_fold);
    for v = 1:num_fold
        train_list = datasets(v).train;
        test_list = datasets(v).test;

        labels = struct('train', [], 'test', []);
        labels.train = double([train_list.label]');
        labels.test = double([test_list.label]');

        % Generate SIFT features
        dict_size = 1024/512;
        sift = encode_sift([train_list.sift]', [test_list.sift]', dict_size);

        % Classify by linear SVM
        c = 10.^[1:-1:-3];
        %acc_list(1, :) = c;
        sift_acc(v+1, :) = linear_classify(sift, labels, c);
    end

    sift_acc = [sum(sift_acc)]
    %save('sift.mat');
end

function acc_list = linear_classify(features, labels, c)
    acc_list = zeros(1, length(c));
    parfor idx = 1:length(c)
        model = train(labels.train, sparse(features.train), ...
                      ['-c ', num2str(c(idx)), ' -q'], 'col');
        [~, acc, ~] = predict(labels.test, ...
                              sparse(features.test), model, '-q', 'col');
        acc_list(idx) = acc(1);
    end
end

function features = encode_sift(train_sift, test_sift, dict_size);
    % Generate codebook by K-means
    [dict, asgn] = vl_ikmeans([train_sift.d], dict_size, 'method', 'elkan');
    train_enc = calc_kmeans_hists(asgn, [train_sift.n], dict_size);

    % Encode testing image by codebook histogram
    asgn = vl_ikmeanspush([test_sift.d], dict);
    test_enc = calc_kmeans_hists(asgn, [test_sift.n], dict_size);

    features.train = train_enc;
    features.test = test_enc;
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

function dataset = parse_image_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    list = [raw{1} num2cell(raw{2})];
    dataset = cell2struct(list, {'path', 'label'}, 2);
    fclose(fd);
end

function descriptor = extract_sift(image)
    descriptors = struct('f', [], 'd', [], 'n', 0);

    % Extract SIFT descriptors
    gray_image = single(rgb2gray(image));
    [frames, local_descriptors] = vl_sift(gray_image);

    descriptor.f = frames;
    descriptor.d = local_descriptors;
    descriptor.n = size(frames, 2);
end

function folds = cross_validation(dataset, num_fold)
    folds(1:num_fold) = struct('train', [], 'test', []);
    categories = unique([dataset.label]);
    for c = 1:length(categories)
        % Select particular category, perform random permutation
        list = dataset([dataset.label] == categories(c));
        len = length(list);
        list = list(randperm(len));

        % Calculate #test_case on each fold
        sample_fold = randsample(num_fold, mod(len, num_fold));
        test_nums(1:num_fold) = floor(len/num_fold);
        test_nums(sample_fold) = floor(len/num_fold)+1;
        test_nums = test_nums - (test_nums==len);  % Ensure #train_instance > 0

        for v = 1:num_fold
            test_list = list(1:test_nums(v));
            train_list = list(test_nums(v)+1:end);

            folds(v).train = [folds(v).train; train_list];
            folds(v).test = [folds(v).test; test_list];
            list = [train_list; test_list];
        end
    end
end

function hists = calc_kmeans_hists(assignment, num_descriptors, dict_size)
    hists = zeros(dict_size, length(num_descriptors));
    offset = cumsum(num_descriptors)-num_descriptors;
    parfor idx = 1:length(num_descriptors)
        v = assignment(offset(idx)+1:offset(idx)+num_descriptors(idx));
        hists(:, idx) = vl_ikmeanshist(dict_size, v);
    end
end

