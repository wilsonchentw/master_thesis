function descriptor = extract_sift(image)
    descriptors = struct('f', [], 'd', [], 'n', 0);

    % Extract SIFT descriptors
    gray_image = single(rgb2gray(image));
    [frames, local_descriptors] = vl_sift(gray_image);

    descriptor.f = frames;
    descriptor.d = local_descriptors;
    descriptor.n = size(frames, 2);
end


