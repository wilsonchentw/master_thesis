function feature_classify(image_list)
    data = parse_image_list(image_list);

    fold = 2;
    datasets = cross_validation(data, fold);
    for idx = 1:fold
        dataset.train.path = datasets(idx).train(:, 2);
        dataset.test.path = datasets(idx).test(:, 2);

        [train_features, test_features] = extract_features(dataset);

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
