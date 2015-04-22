function encode = pooling(alpha, num)
    encode = zeros(size(alpha, 1), length(num));
    for idx = 1:length(num)
        encode(:, idx) = mean(alpha(:, 1:num(idx)), 2);
        alpha = alpha(:, num(idx)+1:end);
    end
end

