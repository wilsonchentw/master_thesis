function extract_features(image_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../libsvm/matlab'));
    addpath(fullfile('../liblinear/matlab'));

    fd = fopen(image_list);
    data = textscan(fd, '%s %d');
    [path, label] = data{:};
    fclose('all');

    sift_descriptors = [];
    for idx = 1:length(path):length(path)
        image = imread(path{idx});
        norm_image = normalize_image(image, [512 512], true);

        % Extract SIFT descriptors
        gray_image = single(rgb2gray(norm_image));
        [f, d] = vl_sift(gray_image);
        sift_descriptors = [sift_descriptors d];
    end

    % K-means clustering
    num_clusters = 10;
    [centers, assignments] = vl_kmeans(single(sift_descriptors), num_clusters);
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
