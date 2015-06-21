function dict = kmeans_dict(vocab, branch, level)
    leaves = branch ^ level;

    tree = vl_hikmeans(vocab, branch, leaves, ...
                       'Method', 'lloyd', 'MaxIters', 400);
    dict = get_tree_center(tree);
    dict = dict(:, end - leaves + 1:end);
end

function centers = get_tree_center(tree)
    queue = tree.sub;
    centers = tree.centers;
    while ~isempty(queue)
        centers = [centers queue(1).centers];
        queue = [queue queue(1).sub];
        queue(1) = [];
    end
end
