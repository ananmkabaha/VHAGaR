function get_dataset_params( dataset )
    if dataset == "mnist"
        w_,h_,k_=28,28,1
        c_ = 10
    elseif dataset == "fmnist"
        w_,h_,k_=28,28,1
        c_ = 10
    elseif dataset == "cifar10"
        w_,h_,k_=32,32,3
        c_ = 10
    end
    return w_,h_,k_,c_
end


