function feature_classify(image_list)
    data = parse_image_list(image_list);

    fold = 2;
    datasets = cross_validation(data, fold);
    for idx = 1:fold
        train_path = datasets(idx).train(:, 2);
        test_path = datasets(idx).test(:, 2);
        train_label = double(cell2mat(datasets(idx).train(:, 1)));
        test_label = double(cell2mat(datasets(idx).test(:, 1)));
        [train_feature, test_feature] = extract_features(train_path, test_path);

        % Classify by liblinear & libsvm
        for c = -3:1
            c_str = num2str(10^c);
            model = train(train_label, train_feature, ['-c ', c_str, ' -q']);
            [guess, acc, ~] = predict(test_label, test_feature, model);
        end
    end;
end

function data = parse_image_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    data = cell2struct(raw, {'path', 'label'}, 2);
    fclose(fd);
end

function lists = cross_validation(data, fold)
    lists(1:fold) = struct('train', [], 'test', []);
    categories = unique(data.label);
    for c = 1:length(categories)
        % Generate list of specific category
        select = (data.label == categories(c));
        list = [num2cell(data.label(select)), data.path(select)];
        %list = cell2struct(list, {'path', 'label'}, 2);
        list_len = length(find(select));

        % Generate #testing_instance and ensure #training_instance > 0
        test_nums = floor(list_len/fold)*ones(1, fold);
        test_nums = test_nums + randerr(1, fold, mod(list_len, fold));
        test_nums = test_nums - (test_nums>=list_len);
        for v = 1:fold
            lists(v).test = [lists(v).test;  list(1:test_nums(v), :)];
            lists(v).train = [lists(v).train; list(test_nums(v)+1:end, :)];
            list = [list(test_nums(v)+1:end, :); list(1:test_nums(v), :)];
        end
    end
end
