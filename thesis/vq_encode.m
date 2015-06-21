function vq = vq_encode(dict, vocabs)
    dict_size = size(dict, 2);

    vq = zeros(dict_size, length(vocabs));
    for idx = 1:length(vocabs)
        dist = vl_alldist(dict, vocabs{idx});
        [~, asgn] = min(dist);
        vq(:, idx) = vl_ikmeanshist(dict_size, asgn);
    end
end
