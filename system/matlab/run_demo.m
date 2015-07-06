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
        [label, score] = rank_candidate(feature, model);


        % Linear scale score => norm_score(1) = 49/50, norm_score = 1/50
        norm_score = score * (atanh(0.96) - atanh(-0.96)) + atanh(-0.96);
        norm_score = tanh(norm_score + atanh(-0.96)) * 0.5 + 0.5;
        norm_score = norm_score / sum(norm_score) * 100;

        %% Direct output probability
        %norm_score = tanh(score) * 0.5 + 0.5;


        % Output
        num_candidate = min(length(label), 5);
        for idx = 1:num_candidate
            name = category_name{label(idx)};
            fprintf('%d %s %.2f ', label(idx), name, norm_score(idx));
        end
        fprintf('\n');

    catch
        fprintf('Error\n');
    end
end
