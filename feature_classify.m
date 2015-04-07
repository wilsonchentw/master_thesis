function feature_classify(image_list)
    [paths, labels] = parse_image_list(image_list);
    [train_lists test_lists] = cross_validation(paths, labels, 5)
end

function [paths, labels] = parse_image_list(image_list)
    fd = fopen(image_list);
    data = textscan(fd, '%s %d');
    [paths labels] = data{:};
    fclose(fd);
end

function [train_lists test_lists] = cross_validation(paths, labels, fold)
    train_lists = cell(1, fold);
    test_lists = cell(1, fold);
    categories = unique(labels);
    for c = 1:length(categories)
        list = paths(labels==categories(c));
        num = length(list);

        % Calculate #testing_instance, and ensure #training_instance > 0
        test_nums = floor(num/fold)*ones(1, fold)+ ([1:fold]<=mod(num, fold));
        test_nums = test_nums - (test_nums==num);
        for v = 1:fold
            test_lists{v} = [test_lists{v}; list(1:test_nums(v))];
            train_lists{v} = [train_lists{v}; list(test_nums(v)+1:end)];
            list = [list(test_nums(v)+1:end); list(1:test_nums(v))];
        end
    end
end
