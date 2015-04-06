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

    % Generate codebook
    [train_paths, train_labels] = train_data{:};
    [dict, train_data] = generate_codebook(train_paths);

    % Project testing data to dictionary
    [test_paths, test_labels] = test_data{:};

    %[~, code] = min(vl_alldist2(sifts, dict));
end

function features = encode_sift(dict, path_list)
end

function [dict, train_data] = generate_codebook(path_list)
    sift_descriptors = {};
    for idx = 1:length(path_list)/length(path_list)*3
        % Normalize image to [h w], cut for central part if crop is true
        image = imread(path_list{idx});
        norm_size = [512 512];
        norm_image = normalize_image(image, norm_size, true);

        % Extract SIFT descriptors
        gray_image = single(rgb2gray(norm_image));
        [f, sift_descriptors{idx}] = vl_sift(gray_image);
    end

    % Generate codebook using K-means clustering on SIFT
    dict_size = 1024;
    dict = single(cell2mat(sift_descriptors));
    [dict, encode] = vl_kmeans(dict, dict_size, 'Initialization', 'plusplus');

    % Generate encoded features of training data
    train_data = [];
    for idx = 1:1:size(sift_descriptors, 2)
        num = size(sift_descriptors{idx}, 2);
        hist = vl_ikmeanshist(dict_size, encode(1:num));

        train_data = [train_data hist]
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
