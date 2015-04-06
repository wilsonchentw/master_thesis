function extract_features(train_list, test_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../libsvm/matlab'));
    addpath(fullfile('../liblinear/matlab'));

    % Parse training/testing path list
    ftrain = fopen(train_list);
    ftest = fopen(test_list);
    train_data = textscan(ftrain, '%s %d');
    test_data = textscan(ftest, '%s %d');
    fclose('all');


    % Generate codebook and encoded training data
    [train_list, train_labels] = train_data{:};
    features = extract_sift(train_list(1:3));
    [dict, train_hists] = generate_codebook(features);

    % Encode testing data
    [test_list, test_labels] = test_data{:};
    features = extract_sift(test_list(1:3));
    test_hists = [];
    for idx = 1:size(features, 2)
        [~, encode] = min(vl_alldist2(double(features{idx}), dict)');
        hist = vl_ikmeanshist(size(dict, 2), encode);
        test_hists = [test_hists; hist'];
    end
end

function sift_descriptors = extract_sift(image_list)
    sift_descriptors = {};
    for idx = 1:length(image_list)
        % Normalize image to [h w], cut for central part if crop is true
        image = imread(image_list{idx});
        norm_size = [512 512];
        norm_image = normalize_image(image, norm_size, true);

        % Extract SIFT descriptors
        gray_image = single(rgb2gray(norm_image));
        [f, sift_descriptors{idx}] = vl_sift(gray_image);
    end
end

function [dict, train_encoded] = generate_codebook(features)
    % Generate dictionary using K-means clustering
    dict_size = 1024;
    dict = double(cell2mat(features));
    [dict, encode] = vl_kmeans(dict, dict_size, 'Initialization', 'plusplus');

    % Generate encoded features of training data
    train_encoded = [];
    for idx = 1:1:size(features, 2)
        num = size(features{idx}, 2);
        hist = vl_ikmeanshist(dict_size, encode(1:num));
        train_encoded = [train_encoded hist];
        encode(1:num) = [];
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
