function lbp = extract_lbp(path)
    level = 3;
    scale = 1 / 2;

    lbp = cell(1, length(path));
    for idx = 1:length(path)
        image = im2single(read_image(path{idx}));

        ds = get_color_lbp(image);
        blur_kernel = fspecial('gaussian', [9 9], 1.6);
        for lv = 2:level
            image = imfilter(image, blur_kernel, 'symmetric');
            image = imresize(image, scale);
            ds = [ds get_color_lbp(image)];
        end

        % Ignore spatial & scale information
        ds = cellfun(@(x) {reshape(x, [], 58)'}, ds);
        for ch = 1:size(ds, 1)
            lbp{idx}{ch, 1} = cell2mat(ds(ch, :));
        end
    end
end

function lbp = get_color_lbp(image)
    image = rgb2lab(image);

    lbp = cell(size(image, 3), 1);
    for ch = 1:size(image, 3)
        lbp{ch} = get_lbp(image(:, :, ch));
    end
end
