function run_demo(path)
    try
        % Setup third party library and script path, then load model
        initialize_environment();
        load('50data_large.mat');

        % Read image, normalize image size and cropping center
        image = read_image(path);

        % Extract color LBP
        ds = get_pyramid_lbp(image);

        % Encode descriptors
        feature = lasso_encode(basis, {ds});
     
        % Homogeneous kernel mapping
        feature = normalize_column(feature, 'L1');
        feature = vl_homkermap(feature, 3, 'kernel', 'kinters', 'gamma', 1);
      
        % Predict label and output
        [label, score] = rank_candidate(feature, model)
    catch
        fprintf('Error\n');
    end
end
