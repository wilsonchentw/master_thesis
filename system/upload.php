<?php 
    header('Content-type: text/html; charset=utf-8');
    $chinese_name = array(
        "Arepas" => "玉米餡餅", 
        "Braised_pork" => "紅燒肉", 
        "Bread" => "麵包", 
        "Buns" => "饅頭", 
        "Chasiu" => "叉燒飯", 
        "Chicken_rice" => "海南雞飯", 
        "Chocolate" => "巧克力", 
        "Corn" => "玉米", 
        "Crab" => "螃蟹", 
        "Croissants" => "牛角麵包", 
        "Curry_chicken" => "咖哩雞", 
        "Curry_rice" => "咖哩飯", 
        "Donut" => "甜甜圈", 
        "Dumplings" => "餃子", 
        "Egg_tart" => "蛋塔", 
        "Fish&Chips" => "魚排與薯片", 
        "Fried_food" => "炸物", 
        "Fried_noodle" => "炒麵", 
        "Fried_Rice" => "炒飯", 
        "Goi_cuon" => "春捲", 
        "Gongbao_chicken" => "宮保雞丁", 
        "Hamburger" => "漢堡", 
        "Hot&Sour_soup" => "酸辣湯", 
        "Ice_cream" => "冰淇淋", 
        "Lasagne" => "千層麵", 
        "Lobster" => "龍蝦", 
        "Mapo_tofu" => "麻婆豆腐", 
        "Noodle_soup" => "擔仔麵", 
        "Omurice" => "蛋包飯", 
        "Peking_duck" => "北平烤鴨", 
        "Pizza" => "披薩", 
        "Popcorn" => "爆米花", 
        "Potato_chip" => "洋芋片", 
        "Poutine" => "起司澆肉汁馬鈴薯條", 
        "Rice_tamale" => "粽子", 
        "Salad" => "沙拉", 
        "Sandwich" => "三明治", 
        "Sashimi" => "生魚片", 
        "Sausage" => "香腸與臘腸", 
        "Shrimp" => "蝦子", 
        "Shumai" => "燒賣", 
        "Som_tam" => "青木瓜絲", 
        "Spaghetti" => "義大利麵", 
        "Steak" => "牛排", 
        "Steamed_sandwich" => "刈包", 
        "Steamed_stuffed_bun" => "小籠包", 
        "Stinky_tofu" => "臭豆腐", 
        "Subway" => "潛艇堡", 
        "Sushi" => "壽司", 
        "Turnip_cake" => "蘿蔔糕"
    );

    $recommend = array(
        "Arepas" => "沙拉、椰奶", 
        "Braised_pork" => "白飯、青菜", 
        "Bread" => "牛奶、茶葉蛋、無糖飲料", 
        "Buns" => "牛奶、蛋、無糖飲料", 
        "Chasiu" => "水果", 
        "Chicken_rice" => "水果、蔬菜", 
        "Chocolate" => "低糖飲料，注意醣類攝取過量", 
        "Corn" => "肉類、蔬菜", 
        "Crab" => "蔬菜、少許肉類", 
        "Croissants" => "牛奶、蛋、無糖飲料", 
        "Curry_chicken" => "白飯、蔬菜、水果", 
        "Curry_rice" => "白飯、少許肉類、蔬菜、水果", 
        "Donut" => "無糖飲料，注意蔬菜肉類攝取不足", 
        "Dumplings" => "酸辣湯、青菜", 
        "Egg_tart" => "低糖飲料，注意澱粉攝取過量", 
        "Fish&Chips" => "沙拉、水果", 
        "Fried_food" => "沙拉、水果", 
        "Fried_noodle" => "青菜、少量肉類、水果", 
        "Fried_Rice" => "青菜、少量肉類、水果", 
        "Goi_cuon" => "湯品", 
        "Gongbao_chicken" => "青菜、水果", 
        "Hamburger" => "可樂，注意蔬菜攝取不足", 
        "Hot&Sour_soup" => "水餃", 
        "Ice_cream" => "低糖紅茶，注意醣類攝取過量", 
        "Lasagne" => "沙拉、蔬菜", 
        "Lobster" => "沙拉、蔬菜", 
        "Mapo_tofu" => "酸辣湯", 
        "Noodle_soup" => "滷味、青菜，注意蔬菜攝取", 
        "Omurice" => "炸物，注意蔬菜攝取", 
        "Peking_duck" => "麵餅、蔬菜", 
        "Pizza" => "可樂、炸雞，注意蔬菜攝取", 
        "Popcorn" => "可樂，注意蔬菜肉類攝取", 
        "Potato_chip" => "飲料，注意所有營養攝取", 
        "Poutine" => "沙拉", 
        "Rice_tamale" => "注意蔬菜攝取", 
        "Salad" => "肉類，注意肉類攝取", 
        "Sandwich" => "牛奶、優酪乳", 
        "Sashimi" => "茶、壽司", 
        "Sausage" => "蔬菜，注意蔬菜攝取", 
        "Shrimp" => "蔬菜、水果，注意蔬菜攝取", 
        "Shumai" => "銀絲卷、叉燒，注意蔬菜攝取", 
        "Som_tam" => "椒麻雞", 
        "Spaghetti" => "沙拉、炸物", 
        "Steak" => "鐵板麵、沙拉", 
        "Steamed_sandwich" => "青蛙撞奶", 
        "Steamed_stuffed_bun" => "炒餅、注意蔬菜攝取", 
        "Stinky_tofu" => "鹹酥雞，注意蔬菜攝取", 
        "Subway" => "低糖飲料", 
        "Sushi" => "無糖綠茶", 
        "Turnip_cake" => "蔥油餅，注意油脂與蔬菜攝取"
    );




    $target_img = $target_dir . basename($_FILES["file"]["name"]);

    $tmp_name = $_FILES["file"]["tmp_name"];
    $image_info = getimagesize($tmp_name);
    if($image_info !== false){
        exec("./run_system.sh $tmp_name", $output);

        $output = explode(" ", implode($output));
        $output = array_chunk($output, 3);

        $ret = array();
        for($i = 0; $i < count($output); $i++){
            $ret[$i] = array(
                'label' => $output[$i][0], 
                'name' => $chinese_name[$output[$i][1]], 
                'score' => $output[$i][2], 
                'recommend' => $recommend[$output[$i][1]]
            );

        }
        echo (json_encode($ret));


        /*
        // Image base64 encoding
        $image_data = base64_encode(file_get_contents($tmp_name));
        $image_mime = $image_info['mime'];
        $image_src = "data:" . $image_mime . ";base64," . $image_data;
        echo '<img src="' . $image_src . '">';
        */
    }
?>
