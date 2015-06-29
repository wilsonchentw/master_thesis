function alpha = linear_blend(t_max, base, label, inst, fold)
    num_category = numel(unique(label));
    num_val = numel(fold.val);
    inst_name = fieldnames(inst);
    num_inst = numel(inst_name);

    label = label(fold.val);
    for idx = 1:num_inst
        inst.(inst_name{idx}) = inst.(inst_name{idx})(fold.val, :);
    end

    % Get validation set prediction
    is_correct = zeros(num_val, num_inst);
    for idx = 1:num_inst
        name = inst_name{idx};
        model = base.(name);
        [guess, acc, prob] = svmpredict(label, inst.(name), model, '-b 1');
        is_correct(:, idx) = (guess == label);
    end

    % Boosted by SAMME, choose best classifier as weak learner
    w = ones(num_val, 1)/num_val;
    ballot = zeros(1, num_inst);
    for t = 1:t_max
        score = w'*is_correct;
        [err, weak] = min(1-score);

        if err == 0
            ballot(weak) = 1;
            break;
        else
            alpha = log((1-err)/err) + log(num_category-1);
            w = w .* exp(alpha*(is_correct(:, weak) == false));
            w = w / sum(w);
            ballot(weak) = ballot(weak) + alpha;
        end
    end

    if abs(sum(ballot)) > eps
        ballot = ballot / sum(ballot);
    end

    alpha = [];
    for idx = 1:num_inst
        alpha.(inst_name{idx}) = ballot(idx);
    end
end

