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
        "Poutine" => "起司澆馬鈴薯條", 
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
        "Arepas" => "注意澱粉攝取", 
        "Braised_pork" => "可搭配白飯、青菜", 
        "Bread" => "注意蔬菜與澱粉攝取", 
        "Buns" => "可搭配牛奶，注意澱粉攝取", 
        "Chasiu" => "飯後水果是不錯的選擇", 
        "Chicken_rice" => "飯後水果是不錯的選擇", 
        "Chocolate" => "注意醣類與熱量攝取過量", 
        "Corn" => "僅有澱粉，請補充其他營養", 
        "Crab" => "蔬菜、少許紅肉", 
        "Croissants" => "可搭配牛奶，注意糖類與澱粉攝取", 
        "Curry_chicken" => "蔬菜攝取可能不足", 
        "Curry_rice" => "飯後水果是不錯的選擇", 
        "Donut" => "醣類攝取過量", 
        "Dumplings" => "可搭配酸辣湯，注意蔬菜攝取", 
        "Egg_tart" => "注意與醣類澱粉攝取過量", 
        "Fish&Chips" => "油脂過量，缺乏蔬菜攝取", 
        "Fried_food" => "油脂過量，缺乏蔬菜攝取", 
        "Fried_noodle" => "注意油脂，下餐可搭配低油食品", 
        "Fried_Rice" => "蔬菜攝取可能不足", 
        "Goi_cuon" => "蔬菜攝取可能不足", 
        "Gongbao_chicken" => "請考慮增加澱粉類主食", 
        "Hamburger" => "蔬菜攝取不足", 
        "Hot&Sour_soup" => "可搭配水餃，注意蔬菜攝取", 
        "Ice_cream" => "醣類攝取過量", 
        "Lasagne" => "可搭配前菜沙拉", 
        "Lobster" => "搭配以蘋果洋蔥的特製醬汁食用", 
        "Mapo_tofu" => "注意鈉含量", 
        "Noodle_soup" => "可搭配燙青菜", 
        "Omurice" => "注意蔬菜攝取量", 
        "Peking_duck" => "可搭配麵餅，但注意蔬菜攝取", 
        "Pizza" => "注意本日攝取熱量", 
        "Popcorn" => "注意澱粉攝取過量", 
        "Potato_chip" => "注意澱粉與鈉攝取過量", 
        "Poutine" => "可搭配前菜沙拉", 
        "Rice_tamale" => "注意蔬菜攝取", 
        "Salad" => "需搭配主食", 
        "Sandwich" => "可搭配牛奶、優酪乳", 
        "Sashimi" => "可搭配壽司，注意蔬菜攝取", 
        "Sausage" => "可搭配主食，注意攝取量", 
        "Shrimp" => "鈉攝取可能過量", 
        "Shumai" => "鈉攝取可能過量", 
        "Som_tam" => "可搭配泰式料理主菜", 
        "Spaghetti" => "可搭配前菜沙拉", 
        "Steak" => "可搭配前菜沙拉", 
        "Steamed_sandwich" => "可搭配青蛙撞奶，注意蔬菜攝取", 
        "Steamed_stuffed_bun" => "可搭配北方麵食，注意蔬菜攝取", 
        "Stinky_tofu" => "注意鈉攝取量", 
        "Subway" => "可搭配低糖飲料", 
        "Sushi" => "可搭配無糖綠茶", 
        "Turnip_cake" => "蔥油餅，注意油脂與澱粉攝取"
    );

    $target_img = $target_dir . basename($_FILES["file"]["name"]);

    $tmp_name = $_FILES["file"]["tmp_name"];
    $image_info = getimagesize($tmp_name);
    if($image_info !== false){
		$tmp_result = uniqid() . '.tmp';
        exec("cmd /c run_system.bat $tmp_name $tmp_result", $output);
		unlink($tmp_result);

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
	}
?>
