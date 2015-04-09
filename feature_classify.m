function feature_classify(image_list)
    % Add libsvm, liblinear, vlfeat library path
    run(fullfile('../vlfeat/toolbox/vl_setup'));
    addpath(fullfile('../liblinear/matlab'));
    addpath(fullfile('../libsvm/matlab'));

    % Parameters setting
    fold = 3;
    c = 10.^[1:-1:-3];
    acc_liblinear = zeros(fold, 5);

    % Preprocessing
    [labels paths] = parse_image_list(image_list);
    datasets = cross_validation(labels, paths, fold);
%{
    parfor v = 1:fold
        features = extract_features(datasets(v).train.path, datasets(v).test.path);
        for log_c = -3:1
            model = num2str(10^log_c);
        end

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
    end;
%}
end

function [labels paths] = parse_image_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    %[paths labels] = raw{:};
    fclose(fd);
end

function datasets = cross_validation(labels, paths, fold)
    datasets(1:fold) = struct('train', struct('label', [], 'path', []), ...
                              'test',  struct('label', [], 'path', []));
    categories = unique(labels);
    for c = 1:length(categories)
        select = (labels == categories(c));
        list = [num2cell(labels(select)) paths(select)];
        len = length(find(select));

        % Generate list of #testing_instance for each fold
        test_nums(1:fold) = floor(len/fold);
        test_nums(randsample(fold, mod(len, fold))) = floor(len/fold)+1;
        test_nums = test_nums - (test_nums==len);  % Ensure #train_instance > 0

        for v = 1:fold
            train = list(test_nums(v)+1:end, :);
            test = list(1:test_nums(v), :);
            datasets(v).train.label = [datasets(v).train.label; 
                                       cell2mat(train(:, 1))];
            datasets(v).test.label = [datasets(v).test.label; 
                                      cell2mat(test(:, 1))];
            datasets(v).train.path = [datasets(v).train.path; train(:, 2)];
            datasets(v).test.path = [datasets(v).test.path; test(:, 2)];
            list = [train; test];
        end
    end
end
