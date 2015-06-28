function image = read_image(path)
    norm_size = [256, 256];
    raw_image = imread(path);
    norm_image = normalize_image(raw_image, norm_size);
    image = im2single(norm_image);
end
