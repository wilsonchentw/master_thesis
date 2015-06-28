function norm_xs = normalize_column(xs, norm_type)
    switch norm_type
        case 'L1'
            norm_xs = xs ./ repmat(sum(xs), size(xs, 1), 1);
        case 'L2'
            norm_xs = xs ./ repmat(sqrt(sum(xs .^ 2)), size(xs, 1), 1);
        otherwise
            fprintf(1, 'Wrong normalize type\n');
            norm_xs = xs;
            return;
    end
end
