function descriptor = extract_gabor(image)
    num_block = [4 4];
    ksize = [15 15];
    theta = [0:5]*pi/6;
    gamma = 1.0;
    lambd = min(ksize)./[1:5];
    sigma = min(ksize)/6;
    psi = 0;

    descriptor = zeros(length(theta)*length(lambd), prod(num_block)*2);
    [thetas, lambds] = ndgrid(theta, lambd);
    for idx = 1:length(theta)*length(lambd)
        theta = thetas(idx);
        lambd = lambds(idx);
        kernel = get_gabor_kernel(ksize, lambd, theta, psi, sigma, gamma);
        %response = imfilter(double(image), kernel);
        %response = filter2(kernel, double(image), 'same');
        response = conv2(double(image)/255, rot90(kernel, 2), 'same');

        blocks = im2col(response, size(image)./num_block, 'distinct');
        magnitude = abs(blocks);
        descriptor(idx, :) = [mean(magnitude) var(magnitude)];
    end
    descriptor = reshape(descriptor, [], 1);
end
