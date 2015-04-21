function dataset = extract_descriptors(dataset)
    % Extract image descriptors
    norm_size = [256 256]/16;
   for idx = 1:length(dataset)
        % Read and preprocessing image
        image = imread(dataset(idx).path);
        norm_image = normalize_image(image, norm_size, true);
        gray_image = rgb2gray(norm_image);

        % Extract SIFT features
        dataset(idx).sift = extract_sift(gray_image);

        % Extract LBP pyramid
        scale = 1/2;
        num_level = 3;
        dataset(idx).lbp = extract_lbp_pyramid(gray_image, scale, num_level);

        % Extract color histogram
        dataset(idx).color = extract_color(norm_image);

        % Extract Gabor filter bank response
        dataset(idx).gabor = extract_gabor(gray_image);
    end
end
