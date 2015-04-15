function acc_list = linear_classify(features, labels, c)
    acc_list = zeros(1, length(c));
    for idx = 1:length(c)
        model = train(labels.train, sparse(features.train), ...
                      ['-c ', num2str(c(idx)), ' -q'], 'col');
        [~, acc, ~] = predict(labels.test, ...
                              sparse(features.test), model, '-q', 'col');
        acc_list(idx) = acc(1);
    end
end


