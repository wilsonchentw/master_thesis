function acc = calculate_accuracy(true_label, rank_label)
    is_acc = (repmat(true_label, size(rank_label, 1), 1) == rank_label);
    acc = mean(is_acc, 2)';
    acc = cumsum(acc);
end
