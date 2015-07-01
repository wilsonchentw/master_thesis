function lbp = extract_lbp(path)
    lbp = cell(2, length(path));
    for idx = 1:length(path)
        image = read_image(path{idx});
        lbp{1, idx} = get_pyramid_lbp(image);

        ds = reshape(lbp{1, idx}, 58, []);
        flip_ds = vl_lbpfliplr(permute(ds, [2 3 1]));
        flip_ds = reshape(permute(flip_ds, [3 1 2]), 174, []);
        lbp{2, idx} = flip_ds;

        if mod(idx, 100) == 0 || idx == length(path)
            fprintf('%4d images done\n');
        end
    end
end
