function descriptor = extract_color(image)
    num_block = [4 4];
    num_bin = 32;
    descriptor = zeros(num_bin, prod(num_block), size(image, 3));

    edges = linspace(0, 256, num_bin+1);
    for color = 1:size(image, 3)
        channel_image = image(:, :, color);
        block_size = size(channel_image)./num_block;
        block = im2col(channel_image, block_size, 'distinct');
        for idx = 1:size(block, 2)
            hist = histcounts(block(:, idx), edges)';
            descriptor(:, idx, color) = hist;
        end
    end
    descriptor = reshape(descriptor, [], 1);
    descriptor = descriptor/sum(descriptor);
end
