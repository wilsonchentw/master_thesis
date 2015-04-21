function descriptor = extract_sift(gray_image)
    % Extract SIFT descriptors
    [frames, local_descriptors] = vl_sift(single(gray_image));
    descriptor = local_descriptors;
end
