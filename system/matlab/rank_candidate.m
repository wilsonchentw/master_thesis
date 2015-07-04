function [label, score] = rank_candidate(feature, model)
    % Generate decision value
    value = model.w * feature;
    if model.nr_class == 2
        value = [value; -value];
    end

    [score, rank] = sort(value, 'descend');
    label = model.Label(rank);
end
