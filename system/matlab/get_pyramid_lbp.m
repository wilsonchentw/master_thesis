function lbp = get_pyramid_lbp(image)
    level = 3;
    scale = 1 / 2;
    blur_kernel = fspecial('gaussian', [9 9], 1.6);

    lbp = get_color_lbp(image);
    for lv = 2:level
        image = imfilter(image, blur_kernel, 'symmetric');
        image = imresize(image, scale);
        lbp = [lbp get_color_lbp(image)];
    end
end


function lbp = get_color_lbp(image)
    lbp = [];
    for ch = 1:size(image, 3)
        ds = get_lbp(image(:, :, ch));
        lbp = [lbp; reshape(ds, [], 58)'];
    end
end


function lbp = get_lbp(image)
    cell_size = 8;
    window_size = 2;

    lbp_cell = vl_lbp(image, cell_size) .^ 2;
    lbp = zeros(size(lbp_cell) - [window_size - 1, window_size - 1, 0]);
    for x = 1:size(lbp, 2)
        for y = 1:size(lbp, 1)
            y_to = y + window_size - 1;
            x_to = x + window_size - 1;

            % Combine several cells into one window
            lbp_block = lbp_cell(y:y_to, x:x_to, :);
            lbp_block = sum(reshape(lbp_block, [], size(lbp_block, 3)));

            % L1-sqrt normalization
            lbp(y, x, :) = lbp_block / sum(lbp_block);
            lbp(y, x, :) = sqrt(lbp(y, x, :));
        end
    end
end
