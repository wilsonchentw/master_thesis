<?php 
    $target_dir = "upload/";
    $target_img = $target_dir . basename($_FILES["file"]["name"]);

    // print_r($_POST);
    // print_r(array_keys($_POST));


    if(isset($_POST["submit"])){
        $image_size = getimagesize($_FILES["file"]["tmp_name"]);
        if($image_size !== false){
            $tmp_name = $_FILES["file"]["tmp_name"];
            exec("./run_demo.sh $tmp_name", $output);
            print_r($output);
        }
    }
?>
