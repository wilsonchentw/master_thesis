function descriptor = extract_lbp_pyramid(image)
    scale = 1/2;
    num_level = 3;
    descriptor = extract_lbp(image);
    blur_kernel = fspecial('gaussian', [9 9], 1.6);
    for level = 2:num_level
        image = imfilter(image, blur_kernel, 'symmetric');
        image = imresize(image, scale);
        descriptor = [descriptor extract_lbp(image)];
    end
end
