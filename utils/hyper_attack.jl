function hyper_attack_hints(m, token, c_tag, c_target, perturbation)

    if perturbation == "max"
        return
    end

    av = JuMP.all_variables(m)
    if isfile("/tmp/fail_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt")
        rm("/tmp/fail_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt")
    else
        file = open("/tmp/booleans_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt", "r")
        data = read(file, String)
        data_array = (rsplit(data,","))
        arr_booleans = []
        for n in eachindex(data_array)
            append!(arr_booleans,parse(Float64, data_array[n]))
        end
        file = open("/tmp/strings_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt", "r")
        data = read(file, String)
        arr_strings = (rsplit(data,","))
        # find indexes
        indexes_ = []
        for n in eachindex(arr_strings)
            ind_to_save = -1
            for k in eachindex(av)
                if arr_strings[n] == JuMP.name(av[k])
                    ind_to_save = deepcopy(k)
                    break
                end
            end
            append!(indexes_,ind_to_save)
        end
        for n in eachindex(arr_strings)
            if indexes_[n]==-1 || arr_booleans[n] == -1
                continue
            end
            set_start_value(av[indexes_[n]],arr_booleans[n])
        end
    end
end

function hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size)
    best_feasible_via_optimization = 0
    pre_time = 0
    if perturbation == "max"
        return best_feasible_via_optimization, pre_time
    end
    pre_time = @elapsed begin
        cmd = Cmd(["python3", "./utils/hyper_attack.py","--dataset", string(dataset), "--source", string(c_tag-1),
        "--target", string(c_target-1), "--token", token_signature, "--model", model_name, "--model_path", model_path*"th", "--perturbation", perturbation,
         "--perturbation_size", create_perturbation_string(perturbation_size)])
        run(cmd)
        best_feasible_via_optimization = read_best_val_via_optimization(c_tag, c_target, token_signature)
    end
    return best_feasible_via_optimization, pre_time
end




