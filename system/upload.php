<?php 
    $target_dir = "upload/";
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
                'name' => $output[$i][1], 
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
