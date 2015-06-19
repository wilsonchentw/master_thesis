function [prefix, label, path] = parse_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    fclose(fd);

    [path, label] = raw{:};
    [~, prefix, ~] = fileparts(image_list);
end
