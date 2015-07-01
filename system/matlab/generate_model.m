function generate_model(image_list)
    setup_3rdparty();

    % Parse input image list
    [~, prefix, ~] = fileparts(image_list);
    [path, label] = parse_list(image_list);
    [category, category_name] = extract_label_name(path, label);

    % Extract descriptors, augment data with flip LBP
    ds = extract_lbp(path);
    ds = reshape(ds', 1, []);
    label = [label; label];

    % Generate basis and encode with LASSO
    basis = generate_basis(ds);
    feature = lasso_encode(basis, ds);

    % Homogeneous kernel mapping
    feature = normalize_column(feature, 'L1');
    feature = vl_homkermap(feature, 3, 'kernel', 'kinters', 'gamma', 1);

    % Generate model
    model = train(double(label), sparse(feature), '-s 1 -c 10 -q', 'col');

    % Save model
    save([prefix, '.mat'], 'basis', 'model', 'category', 'category_name');
end


function [path, label] = parse_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    fclose(fd);

    [path, label] = raw{:};
end


function [category, category_name] = extract_label_name(path, label)
    [category, category_idx, ~] = unique(label);

    category_path = path(category_idx);
    for idx = 1:length(category_path)
        [dirpath, ~, ~] = fileparts(category_path{idx});
        [~, dirname, ~] = fileparts(dirpath);
        category_name{idx} = dirname;
    end

    [category, sort_idx] = sort(category);
    category_name(:) = category_name(sort_idx);
end


function basis = generate_basis(vocabs)
    vocabs = cell2mat(reshape(vocabs, 1, []));

    % Generate descriptor basis
    batch_size = 16384;
    iter = ceil(size(vocabs, 2) / batch_size);
    param = struct('K', 1024, 'lambda', 0.25, 'lambda2', 0, ...
                   'iter', iter, 'mode', 2, 'modeD', 0, ...
                   'batchsize', batch_size, 'modeParam', 0, ...
                   'clean', true, 'numThreads', 4, 'verbose', false);
    fprintf('K = %4d, iter = %4d, batch = %5d\n', ...
            param.K, param.iter, param.batchsize);

    basis = mexTrainDL(vocabs, param);
end
