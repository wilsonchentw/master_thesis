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

