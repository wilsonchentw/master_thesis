function feature_classify(image_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Parameters setting
    num_fold = 3;
    c = 10.^[1:-1:-3];

    % Split data for cross validation
    dataset = parse_image_list(image_list);
    folds = cross_validation(dataset, num_fold);

    % Classify
    train_accs = zeros(num_fold, length(c));
    test_accs = zeros(num_fold, length(c));
    features = extract_features(folds(v));
    for v = 1:num_fold

        % Train by liblinear
        parfor idx = 1:length(c)
            model = train(double(folds(v).train.label), ...
                          sparse(features.train), ...
                          ['-c ', num2str(c(idx)), ' -q'], 'col');
            [~, train_acc, ~] = predict(double(folds(v).train.label), ...
                                        sparse(features.train), model, ...
                                        '-q', 'col');
            [~, test_acc, ~] = predict(double(folds(v).test.label), ...
                                       sparse(features.test), model, ...
                                       '-q', 'col');
            train_accs(v, idx) = train_acc(1);
            test_accs(v, idx) = test_acc(1);
        end
    end;
    train_acc = sum(train_accs)/num_fold
    test_acc = sum(test_accs)/num_fold
end

function dataset = parse_image_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    list = [raw{1} num2cell(raw{2})];
    dataset = cell2struct(list, {'path', 'label'}, 2);
    fclose(fd);
end


function folds = cross_validation(dataset, num_fold)
    folds(1:num_fold) = struct('train', struct('label', [], 'path', []), ...
                               'test',  struct('label', [], 'path', []));
    categories = unique([dataset.label]);
    for c = 1:length(categories)

        % Select particular category
        select = ([dataset.label] == categories(c));
        list = dataset(select);

        % Generate list of #testing_instance for each fold
        len = length(list);
        sample_fold = randsample(num_fold, mod(len, num_fold));
        test_nums(1:num_fold) = floor(len/num_fold);
        test_nums(sample_fold) = floor(len/num_fold)+1;
        test_nums = test_nums - (test_nums==len);  % Ensure #train_instance > 0

        for v = 1:num_fold
            test_list = list(1:test_nums(v));
            train_list = list(test_nums(v)+1:end);
            folds(v).train.label = [folds(v).train.label; [train_list.label]'];
            folds(v).test.label = [folds(v).test.label; [test_list.label]'];
            folds(v).train.path = [folds(v).train.path; [{train_list.path}]'];
            folds(v).test.path = [folds(v).test.path; [{test_list.path}]'];
            list = [train_list; test_list];
        end
    end
end
