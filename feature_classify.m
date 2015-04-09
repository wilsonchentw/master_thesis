function feature_classify(image_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Parameters setting
    num_fold = 3;
    c = 10.^[1:-1:-3];
    acc_liblinear = zeros(num_fold, 5);

    % Preprocessing
    dataset = parse_image_list(image_list);
    folds = cross_validation(dataset, num_fold);
    parfor v = 1:num_fold
        tic
        features = extract_features(folds(v).train.path, folds(v).test.path);
        toc
        for log_c = -3:1
            model = num2str(10^log_c);
        end
%{
        %train_path = datasets(idx).train(:, 2);
        %test_path = datasets(idx).test(:, 2);
        %train_label = double(cell2mat(datasets(idx).train(:, 1)));
        %test_label = double(cell2mat(datasets(idx).test(:, 1)));
        %[train_insts, test_insts] = extract_features(train_path, test_path);

        %% Classify by liblinear & libsvm
        %for c = -3:1
        %    c_str = num2str(10^c);
        %    model = train(train_label, train_insts, ['-c ', c_str, ' -q']);
        %    [guess, acc, ~] = predict(test_label, test_insts, model);
        %end
%}
    end;
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
