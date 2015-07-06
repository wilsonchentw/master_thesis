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
        "Som_tam" => "春捲", 
        "Spaghetti" => "義大利麵", 
        "Steak" => "牛排", 
        "Steamed_sandwich" => "刈包", 
        "Steamed_stuffed_bun" => "小籠包", 
        "Stinky_tofu" => "臭豆腐", 
        "Subway" => "潛艇堡", 
        "Sushi" => "壽司", 
        "Turnip_cake" => "蘿蔔糕"
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
                'score' => $output[$i][2]
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
