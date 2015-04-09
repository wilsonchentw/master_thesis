function features = extract_features(train_images, test_images)
    features = struct('train', [], 'test', []);
    descriptors(1:2) = struct('sift', []);

    % Extract image descriptors
    norm_size = [64 64];
    descriptors(1).sift = extract_sift(train_images, norm_size);
    descriptors(2).sift = extract_sift(test_images, norm_size);
    %size()

    % Generate codebook and encode training image
    dict_size = 1024/256;
    [dict, assignment] = vl_ikmeans([descriptors(1).sift.d], dict_size);
    %[dict, assignment] = vl_kmeans(double([descriptors(1).sift.d]), ...
    %                               dict_size, 'Initialization', 'plusplus');
    features.train = calc_hists(assignment, [descriptors(1).sift.n], dict_size);

    % Encode training image and testing image
    assignment = vl_ikmeanspush([descriptors(2).sift.d], dict);
    %[~, assignment] = min(vl_alldist2(dict, double(descriptors(2).sift.d)));
    features.test = calc_hists(assignment, [descriptors(2).sift.n], dict_size);

    % Write in libsvm format
    %libsvmwrite(train_dat, double(train_labels), sparse(train_encode));
    %libsvmwrite(test_dat, double(test_labels), sparse(test_encode));
end

function descriptors = extract_sift(image_list, norm_size)
    descriptors(length(image_list)) = struct('f', [], 'd', [], 'n', 0);
    parfor idx = 1:length(image_list)
        image = imread(image_list{idx});
        norm_image = normalize_image(image, norm_size, true);
        gray_image = single(rgb2gray(norm_image));

        % Extract SIFT descriptors
        [frames, local_descriptors] = vl_sift(gray_image);
        descriptors(idx).f = frames;
        descriptors(idx).d = local_descriptors;
        descriptors(idx).n = 0;
    end
end

function hists = calc_hists(assignment, num_descriptors, dict_size)
    hists = zeros(dict_size, length(num_descriptors));
    offset = cumsum(num_descriptors)-num_descriptors;
    parfor idx = 1:length(num_descriptors)
        v = assignment(offset(idx):offset(idx)+num_descriptors(idx)-1);
        hists(:, idx) = vl_ikmeanshist(dict_size, v);
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
        resized_image = imresize(image, scale);
        norm_image = resized_image(y, x, :);
    end
end
