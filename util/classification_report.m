function report = classification_report(true_label, rank_label)
    num_inst = length(true_label);
    report = struct('accuracy', []);

    % Only store top-1 confusion matrix, precision, and recall rate
    [cm, order] = confusionmat(true_label, rank_label(1, :));
    true_hist = histcounts(true_label, 'BinMethod', 'integers');
    pred_hist = sum(cm);

    tp = diag(cm)';
    fp = (pred_hist - tp);
    fn = true_hist - tp;
    tn = num_inst - (tp + fp + fn);

    report.confusion_matrix = cm;
    report.precision = tp ./ (tp + fp);
    report.recall = tp ./ (tp + fn);


    % Calculate top-N accuracy
    for rank = 1:size(rank_label, 1)
        is_correct = true_label == rank_label(rank, :);
        report.accuracy(rank) = sum(is_correct) / num_inst;
    end
    report.accuracy = cumsum(report.accuracy);
end



