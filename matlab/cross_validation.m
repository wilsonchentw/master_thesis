function folds = cross_validation(dataset, num_fold)
    folds(1:num_fold) = struct('train', [], 'test', []);
    categories = unique([dataset.label]);
    for c = 1:length(categories)
        % Select particular category, perform random permutation
        list = find([dataset.label] == categories(c))';
        len = length(list);
        list = list(randperm(len));

        % Calculate #test_case on each fold
        sample_fold = randsample(num_fold, mod(len, num_fold));
        test_nums(1:num_fold) = floor(len/num_fold);
        test_nums(sample_fold) = floor(len/num_fold)+1;
        test_nums = test_nums - (test_nums==len);  % Ensure #train_instance > 0

        for v = 1:num_fold
            test_list = list(1:test_nums(v));
            train_list = list(test_nums(v)+1:end);

            folds(v).train = [folds(v).train; train_list];
            folds(v).test = [folds(v).test; test_list];
            list = [train_list; test_list];
        end
    end
end


