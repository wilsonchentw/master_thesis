function hists = kmeans_hists(assignment, num_descriptors, dict_size)
    hists = zeros(dict_size, length(num_descriptors));
    offset = cumsum(num_descriptors)-num_descriptors;
    for idx = 1:length(num_descriptors)
        v = assignment(offset(idx)+1:offset(idx)+num_descriptors(idx));
        hists(:, idx) = vl_ikmeanshist(dict_size, v);
    end
end


