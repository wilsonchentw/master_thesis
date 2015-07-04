function basis = generate_basis(vocabs)
    vocabs = cell2mat(reshape(vocabs, 1, []));

    % Generate descriptor basis
    batch_size = 16384;
    iter = ceil(size(vocabs, 2) / batch_size);
    param = struct('K', 1024, 'lambda', 0.25, 'lambda2', 0, ...
                   'iter', iter, 'mode', 2, 'modeD', 0, ...
                   'batchsize', batch_size, 'modeParam', 0, ...
                   'clean', true, 'numThreads', 4, 'verbose', false);
    fprintf('K = %4d, iter = %4d, batch = %5d\n', ...
            param.K, param.iter, param.batchsize);

    basis = mexTrainDL(vocabs, param);
end

