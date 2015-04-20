function descriptor = extract_gabor(image)
    num_block = [4 4];
    ksize = [9 9];
    theta = [0:5]*pi/6;
    gamma = 1.0;
    lambd = min(ksize)./[1:5];
    sigma = min(ksize)/6;
    psi = 0;

    descriptor = zeros(length(theta)*length(lambd), prod(num_block)*2);
    gray_image = double(rgb2gray(image));
    [thetas, lambds] = ndgrid(theta, lambd);
    for idx = 1:length(theta)*length(lambd)
        theta = thetas(idx);
        lambd = lambds(idx);
        kernel = get_gabor_kernel(ksize, lambd, theta, psi, sigma, gamma);

        response = imfilter(gray_image, kernel);
        blocks = im2col(response, size(gray_image)./num_block, 'distinct');
        magnitude = sqrt(blocks.*conj(blocks));
        descriptor(idx, :) = [mean(magnitude) var(magnitude)];
    end
    descriptor = reshape(descriptor, [], 1);
end

function filter = get_gabor_kernel(ksize, lambd, theta, psi, sigma, gamma)
    y = -floor(ksize(1)/2):floor(ksize(1)/2);
    x = -floor(ksize(2)/2):floor(ksize(2)/2);
    [x, y] = meshgrid(x, y);
    rx = x*cos(theta) + y*sin(theta);
    ry = -x*sin(theta) + y*cos(theta);

    window = exp((-rx.^2-(gamma*ry).^2)/(sigma^2*2));
    sinusoid = exp(j*(2*pi*rx/lambd + psi));
    filter = window.*sinusoid;
end
