function [path, label, label_name] = parse_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    [path, label] = raw{:};
    fclose(fd);

    % Get unique directory name
    [category, category_idx, ~] = unique(label);
    category_path = cellfun(@(x) {fileparts(x)}, path(category_idx));
    for idx = 1:length(category_path)
        [~, dirname, ~] = fileparts(category_path{idx});
        label_name{idx} = dirname;
    end

    [~, sort_idx] = sort(category);
    label_name(:) = label_name(sort_idx);
end
