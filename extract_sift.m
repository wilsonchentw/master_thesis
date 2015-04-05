function extract_sift(image_list)
    fd = fopen(image_list);
    data = textscan(fd, '%s %d');
    [path, label] = data{:};

    for idx = 1:length(path)/100:length(path)
        image = imread(path{idx});
        norm_image = normalize_image(image, [512 512], true);
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
