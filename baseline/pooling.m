function encode = pooling(alpha, num)
    encode = zeros(size(alpha, 1), length(num));
    rightmost = cumsum(num);
     for idx = 1:length(num)
        from = rightmost(idx)-num(idx)+1;
        to = rightmost(idx);
        encode(:, idx) = mean(alpha(:, from:to), 2);
    end
end
%{
function encode = pooling(alpha, num)
    encode = zeros(size(alpha, 1), length(num));
    for idx = 1:length(num)
        encode(:, idx) = mean(alpha(:, 1:num(idx)), 2);
        %alpha = alpha(:, num(idx)+1:end);
        alpha(1:num(idx)) = [];
    end
end
%}
