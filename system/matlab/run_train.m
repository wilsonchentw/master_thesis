function run_train(image_list)
    initialize_environment();

    % Parse input image list
    [~, prefix, ~] = fileparts(image_list);
    [path, label] = parse_list(image_list);
    [category, category_name] = extract_label_name(path, label);

    % Extract descriptors, augment data with flip LBP
    ds = reshape(extract_lbp(path)', 1, []);
    label = [label; label];

    % Generate descriptor basis and encode descriptors
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
