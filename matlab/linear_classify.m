function acc_list = linear_classify(label, feature, fold, c)
    acc_list = zeros(1, length(c));
    train_inst =  sparse(feature(:, fold.train));
    train_label = double(label(fold.train)');
    test_inst = sparse(feature(:, fold.test));
    test_label = double(label(fold.test)');

    for idx = 1:length(c)
        c_str = sprintf('%.10f\n', c(idx));
        model = train(train_label, train_inst, ['-c ', c_str, ' -q'], 'col');
        [g, acc, p] = predict(test_label, test_inst, model, '-q', 'col');
        acc_list(idx) = acc(1);
    end
end


