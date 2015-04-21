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
