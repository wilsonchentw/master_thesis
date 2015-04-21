function acc_list = linear_classify(label, feature, fold)
    c = 10.^[2:-1:-7];
    acc_list = zeros(1, length(c));

    train_inst = sparse(feature(:, fold.train)');
    test_inst = sparse(feature(:, fold.test)');
    train_label = double(label(fold.train)');
    test_label = double(label(fold.test)');

    for idx = 1:length(c)
        model = train(train_label, train_inst, ['-c ', num2str(c(idx)), ' -q']);
        [guess, acc, prob] = predict(test_label, test_inst, model, '-q');
        acc_list(idx) = acc(1);
    end
end


