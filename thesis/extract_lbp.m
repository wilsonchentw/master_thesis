function lbp = extract_lbp(path)
    level = 3;
    scale = 1 / 2;

    lbp = cell(2, length(path));
    for idx = 1:length(path)
        image = im2single(read_image(path{idx}));

        ds = get_color_lbp(image);
        blur_kernel = fspecial('gaussian', [9 9], 1.6);
        for lv = 2:level
            image = imfilter(image, blur_kernel, 'symmetric');
            image = imresize(image, scale);
            ds = [ds get_color_lbp(image)];
        end

        % Calculate flip LBP and merge pyramid descriptors
        flip_ds = cellfun(@vl_lbpfliplr, ds, 'UniformOutput', false);
        flip_ds = cell2mat(cellfun(@(x) {reshape(x, [], 58)'}, flip_ds));
        ds = cell2mat(cellfun(@(x) {reshape(x, [], 58)'}, ds));

        % Merge channel together or not
        merge_channel = true;
        if merge_channel
            lbp{1, idx} = {ds};
            lbp{2, idx} = {flip_ds};
        else
            channel_dim = repmat([58], 1, size(ds, 1) / 58);
            lbp{1, idx} = mat2cell(ds, channel_dim, [size(ds, 2)]);
            lbp{2, idx} = mat2cell(flip_ds, channel_dim, [size(ds, 2)]);
        end
    end
end

function lbp = get_color_lbp(image)
    %image = rgb2lab(image);

    lbp = cell(size(image, 3), 1);
    for ch = 1:size(image, 3)
        lbp{ch} = get_lbp(image(:, :, ch));
    end
end
