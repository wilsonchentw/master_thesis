function descriptor = extract_lbp(gray_image)
    step_size = 8;
    window_size = 16;

    image = single(gray_image);
    block = @(y, x) image(y:y+window_size-1, x:x+window_size-1);
    y = [1:step_size:size(image, 1)-window_size+1];
    x = [1:step_size:size(image, 2)-window_size+1];
    [ys, xs] = meshgrid(y, x); 
    descriptor = zeros(58, length(y)*length(x));
    for idx = 1:length(x)*length(y)
        patch = block(ys(idx), xs(idx));
        patch_lbp = vl_lbp(patch, window_size);
        descriptor(:, idx) = permute(patch_lbp, [3 1 2]);
    end
end