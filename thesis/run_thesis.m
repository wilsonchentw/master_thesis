function run_thesis(image_list)
    setup_3rdparty(pwd)
    [prefix, label, path] = parse_list(image_list);

    %sift = extract_sift(path);
    %lbp = extract_pyramid_lbp(path);
    color = extract_color(path);

end

function setup_3rdparty(root_dir)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile(root_dir, '../vlfeat/toolbox/vl_setup'));
    addpath(fullfile(root_dir, '../liblinear/matlab'));
    addpath(fullfile(root_dir, '../libsvm/matlab'));

    % Add SPAMS(SPArse Modeling Software) path
    addpath(fullfile(root_dir, '../spams/spams-matlab/build'));
    addpath(fullfile(root_dir, '../spams/spams-matlab/test_release'));
    addpath(fullfile(root_dir, '../spams/spams-matlab/src_release'));
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

function sift = extract_sift(path)
    sift = cell(1, length(path));
    for idx = 1:length(path)
        image = read_image(path{idx});
        gray_image = rgb2gray(image);

        % Extract SIFT descriptors
        [frames, descriptors] = vl_sift(single(gray_image));
        descriptors = double(descriptors) / 255.0;
        sift{idx} = descriptors;
    end
end

function lbp = extract_pyramid_lbp(path)
    level = 3;
    scale = 1 / 2;

    lbp = cell(level, length(path));
    blur_kernel = fspecial('gaussian', [9 9], 1.6);
    for idx = 1:length(path)
        image = read_image(path{idx});
        gray_image = rgb2gray(image);

        lbp{1, idx} = get_lbp(single(gray_image));
        for lv = 2:level
            gray_image = imfilter(gray_image, blur_kernel, 'symmetric');
            gray_image = imresize(gray_image, scale);
            lbp{lv, idx} = get_lbp(single(gray_image));
        end
    end
end

function lbp = get_lbp(image)
    step_size = 8;
    window_size = 16;

    lbp_window = vl_lbp(image, window_size);
    lbp_overlap = vl_lbp(image(step_size:end, step_size:end), window_size);

    lbp = [reshape(lbp_window, [], 58)', reshape(lbp_overlap, [], 58)'];
end

function color = extract_color(path)
    color = cell(1, length(path));
    %{
    for idx = 1:length(path)
        image = read_image(path{idx});
        gray_image = rgb2gray(image);

        % Extract SIFT descriptors
        [frames, descriptors] = vl_sift(single(gray_image));
        descriptors = double(descriptors) / 255.0;
        sift{idx} = descriptors;
    end
    %}
end
