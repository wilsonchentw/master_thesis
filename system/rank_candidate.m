function label = rank_candidate(feature, model)
    % Generate decision value
    value = model.w * feature;
    if model.nr_class == 2
        value = [value; -value];
    end

    [~, rank] = sort(value, 'descend');
    label = model.Label(rank);
end
