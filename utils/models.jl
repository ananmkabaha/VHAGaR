
function get_nn(model_path, model_name, w, h, k, c, dataset)

    if model_name == "FC0"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif model_name == "FC1"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 50
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif model_name == "CNN0"
        is_conv = true
        stride_to_use_1 = 4
        stride_to_use_2 = 3
        conv_filters = 3
        conv_filters2 = 3
        flatten_num = 12
        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))), )
        conv1 = get_conv_params(dict1, "conv1", (4, 4, 1, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_conv_params(dict1, "conv2", (3, 3, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, c))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(), Flatten([1, 2, 3, 4]), fc1, ],"nn", )
    elseif model_name == "CNN1"
        is_conv = true
        stride_to_use_1 = 3
        stride_to_use_2 = 3
        conv_filters = 6
        conv_filters2 = 6
        flatten_num = 54
        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        conv1 = get_conv_params(dict1, "conv1", (4, 4, k, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_conv_params(dict1, "conv2", (3, 3, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, 10))
        fc2 = get_matrix_params(dict1, "fc2", (10, c))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2, ],"nn", )
    elseif model_name == "CNN2"
        is_conv = true
        stride_to_use_1 = 1
        stride_to_use_2 = 3
        conv_filters = 3
        conv_filters2 = 3
        flatten_num = 192
        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        conv1 = get_conv_params(dict1, "conv1", (4, 4, k, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_conv_params(dict1, "conv2", (3, 3, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, 10))
        fc2 = get_matrix_params(dict1, "fc2", (10, c))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2, ],"nn", )
    end

    return nn
end