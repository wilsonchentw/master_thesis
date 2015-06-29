function demo(image_path, dataset)

    % Setup up third-party library and load dataset
    setup_3rdparty();
    load(dataset);

    % Read image, normalize image size and cropping center
    image = read_image(image_path);

    % Extract color LBP
    ds = get_pyramid_lbp(image);

    % Encode descriptors
    feature = lasso_encode(basis, {ds});
 
    % Homogeneous kernel mapping
    feature = normalize_column(feature, 'L1');
    feature = vl_homkermap(feature, 3, 'kernel', 'kinters', 'gamma', 1);
  
    % Predict label and output
    label = rank_candidate(feature, model);
    category_idx = arrayfun(@(x) find(category == x), label);

    % Display output
    category_rank = category_name(category_idx)';
    disp(category_rank{1});

end
