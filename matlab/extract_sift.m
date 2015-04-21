function descriptor = extract_sift(gray_image)
    descriptors = struct('f', [], 'd', [], 'n', 0);

    % Extract SIFT descriptors
    [frames, local_descriptors] = vl_sift(single(gray_image));

    descriptor.f = frames;
    descriptor.d = local_descriptors;
end
