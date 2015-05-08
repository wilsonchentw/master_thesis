function dataset = parse_image_list(image_list)
    fd = fopen(image_list);
    raw = textscan(fd, '%s %d');
    list = [raw{1} num2cell(raw{2})];
    dataset = cell2struct(list, {'path', 'label'}, 2);
    fclose(fd);
end


