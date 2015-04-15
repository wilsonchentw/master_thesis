function descriptor = extract_lbp(image)
    descriptor = struct('d', [], 'n', 0);

    gray_image = single(rgb2gray(image));
    d = vl_lbp(gray_image, 8);
    descriptor.d = reshape(permute(d, [3 1 2]), size(d, 3), []);
    descriptor.d = [descriptor.d; 1-sum(descriptor.d)];
    descriptor.n = size(descriptor.d, 2);
end


