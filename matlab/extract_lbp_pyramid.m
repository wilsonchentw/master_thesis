function descriptor = extract_lbp_pyramid(image, scale, num_level)
    descriptor = cell(1, num_level);

    descriptor(1) = {extract_lbp(image)};
    blur_kernel = fspecial('gaussian', [9 9], 1.6);
    for level = 2:num_level
        image = imfilter(image, blur_kernel, 'symmetric');
        image = imresize(image, scale);
        descriptor(level) = {extract_lbp(image)};
    end
end
