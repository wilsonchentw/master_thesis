function run_demo(path)
    try
        % Setup third party library and script path, then load model
        initialize_environment();
        load('50data.mat');

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
        [label, score] = rank_candidate(feature, model);

        % Linear scale score => norm_score(1) = 49/50, norm_score(0) = 1/50
        anchor = ((1.0 - 1.0 / length(label)) - 0.5) * 2.0;
        norm_score = score * (atanh(anchor) - atanh(-anchor)) + atanh(-anchor);
        norm_score = tanh(norm_score + atanh(-anchor)) * 0.5 + 0.5;
        norm_score = norm_score / sum(norm_score) * 100;

        % Output
        num_candidate = min(length(label), 5);
        for idx = 1:num_candidate
            name = category_name{label(idx)};
            fprintf('%d %s %.2f ', label(idx), name, norm_score(idx));
        end
        fprintf('\n');

    catch
        fprintf('%d Error 100.0\n', length(label) + 1);
    end
end
