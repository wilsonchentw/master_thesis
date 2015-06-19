function image = read_image(path)
    norm_size = [256, 256];

    % TODO: Check if image is valid 3-channel image
    raw_image = imread(path);
    image = normalize_image(raw_image, norm_size);
end
