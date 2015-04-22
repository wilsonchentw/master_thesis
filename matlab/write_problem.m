function write_problem(filename, label, feature, fold)
    train_inst =  sparse(feature(:, fold.train)');
    train_label = double(label(fold.train)');
    test_inst = sparse(feature(:, fold.test)');
    test_label = double(label(fold.test)');

    libsvmwrite([filename, '_train.dat'], train_label, train_inst);
    libsvmwrite([filename, '_test.dat'], test_label, test_inst);
end
