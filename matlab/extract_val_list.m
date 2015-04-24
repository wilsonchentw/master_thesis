function val_list = extract_val_list(label, train_list, num_fold)
    categories = unique(label);
    val_list = [];
    for c = 1:length(categories)
        category_list = train_list(label(train_list)==categories(c));
        category_len = length(category_list);
        val_num = floor(category_len/num_fold);
        if val_num == 0 && category_len > 1
            val_num = 1;
        end

        val_samples = category_list(randsample(category_len, val_num));
        val_list = [val_list val_samples];
    end
end
