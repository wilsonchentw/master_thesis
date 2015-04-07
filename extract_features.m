function extract_features(train_list, test_list, train_dat, test_dat)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Parse training/testing path list
    [train_images, train_labels] = parse_image_list(train_list);
    [test_images, test_labels] = parse_image_list(test_list);
    train_images = train_images(1:5); train_labels = train_labels(1:5);
    test_images = test_images(1:3);   test_labels = test_labels(1:3);

    % Generate codebook and encode training data
    norm_size = [64 64];
    dict_size = 1024/128;
    features = extract_sift(train_images, norm_size);
    [dict, train_encode] = generate_codebook(features, dict_size);

    % Encode testing data with codebook
    features = extract_sift(test_images, norm_size);
    [~, assignment] = min(vl_alldist2(dict, double(features.descriptors)));
    test_encode = instances_histogram(features.num, assignment, dict_size);

    % Write in libsvm format
    libsvmwrite(train_dat, double(train_labels), sparse(train_encode));
    libsvmwrite(test_dat, double(test_labels), sparse(test_encode));
end

function [paths, labels] = parse_image_list(image_list)
    fd = fopen(image_list);
    data = textscan(fd, '%s %d');
    [paths labels] = data{:};
    fclose(fd);
end

function encodes = instances_histogram(num_descriptors, assignment, dict_size)
    encodes = [];
    for idx = 1:length(num_descriptors)
        encode = assignment(1:num_descriptors(idx));
        hist = vl_ikmeanshist(dict_size, encode)';
        encodes = [encodes; double(hist)/sum(hist)];
        assignment(1:num_descriptors(idx)) = [];
    end
end

function [dict, train_encode] = generate_codebook(features, dict_size)
    % Generate dictionary using K-means clustering
    vocabs = double(features.descriptors);
    [dict, asgn] = vl_kmeans(vocabs, dict_size, 'Initialization', 'plusplus');

    % Generate encoded features of training data
    train_encode = instances_histogram(features.num, asgn, dict_size);
end

function features = extract_sift(image_list, norm_size)
    features.num = [];
    features.descriptors = [];
    for idx = 1:length(image_list)
        % Normalize image to [h w], cut for central part if crop is true
        image = imread(image_list{idx});
        norm_image = normalize_image(image, norm_size, true);

        % Extract SIFT descriptors
        gray_image = single(rgb2gray(norm_image));
        [f, d] = vl_sift(gray_image);
        features.num = [features.num size(d, 2)];
        features.descriptors = [features.descriptors d];
    end
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
        norm_image = imresize(image, scale);
        norm_image = norm_image(y, x, :);
    end
end
