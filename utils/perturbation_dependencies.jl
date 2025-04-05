function perturbation_init_deps(phi_dep, perturbation, perturbation_size)
    if perturbation == "occ"
        row_index = Int(perturbation_size[1])
        column_index = Int(perturbation_size[2])
        occlusion_width = Int(perturbation_size[3])
        phi_dep .= 0
        phi_dep[1, row_index:row_index+occlusion_width-1, column_index:column_index+occlusion_width-1, :] .= -1
    elseif perturbation == "brightness"
        b = perturbation_size[1]
        if b>0
            phi_dep .= 1
        elseif b<0
            phi_dep .= -1
        elseif b==0
            phi_dep .= 0
        end
    elseif perturbation == "patch"
        row_index = Int(perturbation_size[2])
        column_index = Int(perturbation_size[3])
        occlusion_width = Int(perturbation_size[4])
        phi_dep .= 0
        phi_dep[1, row_index:row_index+occlusion_width-1, column_index:column_index+occlusion_width-1, :] .= NaN
    elseif perturbation == "translation"
        translation_down = perturbation_size[1]
        translation_right = perturbation_size[2]
        phi_dep[:,1:translation_down,:,:] .= -1
        phi_dep[:,:,1:translation_right,:] .= -1
    elseif perturbation == "contrast"
        c = perturbation_size[1]
        if c>0
            phi_dep .= 1
        elseif c<0
            phi_dep .= -1
        elseif c==0
            phi_dep .= 0
        end
    end
end

function dep_propagation(layer, phi_dep)
    ls =deepcopy(layer)
    if occursin("Linear", string(typeof(layer)))
        ls.matrix = abs.(sign.(ls.matrix))
    elseif occursin("Conv", string(typeof(layer)))
        ls.filter = abs.(sign.(ls.filter))
    end
    ls.bias = 0.0 .*(ls.bias)
    t = abs.(phi_dep) |> ls
    ls =deepcopy(layer)
    if occursin("Linear", string(typeof(layer)))
        ls.matrix = sign.(ls.matrix)
    elseif occursin("Conv", string(typeof(layer)))
        ls.filter = sign.(ls.filter)
    end
    ls.bias = 0.0 .*(ls.bias)
    o = phi_dep |> ls
    r = (o./t)
    r[((r .< 1) .& (r .> -1)) ] .= NaN
    r[ (t.==0) .& (o.==0)] .= 0
    return r
end

function layers_number(nn)
    cnt = 0
    for l in nn.layers
        if occursin("ReLU", string(typeof(l)))
            cnt += 1
        end
    end
    return cnt
end

function dep_additional(m, layers_n, layer, phi_dep, phi_dep_l, perturbation, perturbation_size, activation_cnt)
    if (perturbation == "brightness") & (activation_cnt == 1)
        b = perturbation_size[1]
        ls =deepcopy(layer)
        ls.bias = 0.0 .* (ls.bias)
        phi_dep_c = sign.(fill(b, size(phi_dep)) |> layer)
        phi_dep_l[phi_dep_l .==NaN] .= phi_dep_c[phi_dep_l .==NaN]
    elseif (perturbation == "contrast") & (activation_cnt == 1) & all(layer.bias .== 0)
        layers_n = layers_number(nn)
        av = JuMP.all_variables(m)
        c = perturbation_size[1]
        if length(size(phi_dep)) == 4
            phi_dep_c = phi_dep |> Flatten([1, 2, 3, 4])
        else
            phi_dep_c = phi_dep
        end
        for n in 1:size(phi_dep_c)[1]
            dep = phi_dep_c[n]
            layers_info_dict[activation_cnt,n]
            if haskey(layers_info_dict,(activation_cnt,n)) && haskey(layers_info_dict,(activation_cnt+layers_n,n))
                if (dep == NaN)
                    @constraint(m,av[ind_o+1]==(1+av[1])*av[ind_p+1])
                    @constraint(m,av[ind_o+2]==av[ind_p+2])
                end
            end
        end
    end
    phi_dep = phi_dep_l
    return phi_dep
end

function encode_dependencies(m, layers_n, phi_dep, activation_cnt, non_equality_tolerance = 1e-4)
    av = JuMP.all_variables(m)
    if length(size(phi_dep)) == 4
        phi_dep_c = phi_dep |> Flatten([1, 2, 3, 4])
    else
        phi_dep_c = phi_dep
    end
    for n in 1:size(phi_dep_c)[1]
        dep = phi_dep_c[n]
        if haskey(layers_info_dict,(activation_cnt,n)) && haskey(layers_info_dict,(activation_cnt+layers_n,n))
            u_o, l_o, ind_o = layers_info_dict[activation_cnt,n]
            u_p, l_p, ind_p = layers_info_dict[activation_cnt+layers_n,n]
            if dep == 0
                @constraint(m,av[ind_o+1]==av[ind_p+1])
                @constraint(m,av[ind_o+2]==av[ind_p+2])
            elseif dep == 1
                @constraint(m,av[ind_o+1]<=av[ind_p+1])
                @constraint(m,av[ind_o+2]<=av[ind_p+2])
            elseif dep == -1
                @constraint(m,av[ind_o+1]>=av[ind_p+1])
                @constraint(m,av[ind_o+2]>=av[ind_p+2])
            else
                if reuse_bounds_conf.is_reuse_bounds_and_deps == false
                    if l_o >=u_p
                        @constraint(m,av[ind_o+1]>=av[ind_p+1])
                        @constraint(m,av[ind_o+2]>=av[ind_p+2])
                        phi_dep_c[n] = -1
                    elseif l_p>=u_o
                        @constraint(m,av[ind_o+1]<=av[ind_p+1])
                        @constraint(m,av[ind_o+2]<=av[ind_p+2])
                        phi_dep_c[n] = 1
                    else
                        l_diff = Inf
                        u_diff = Inf
                        if (u_o>=u_p) .& (l_o>=l_p)
                            v_obj = @variable(m)
                            @constraint(m, v_obj == av[ind_o+1]-av[ind_p+1])
                            @objective(m, Min, v_obj)
                            set_optimizer_attribute(m, "Cutoff", 0)
                            optimize!(m)
                            l_diff = JuMP.objective_bound(m)
                        end
                        if (u_p>=u_o) .& (l_p>=l_o)
                            v_obj = @variable(m)
                            @constraint(m, v_obj == av[ind_o+1]-av[ind_p+1])
                            @objective(m, Max, v_obj)
                            set_optimizer_attribute(m, "Cutoff", 0)
                            optimize!(m)
                            u_diff = JuMP.objective_bound(m)
                        end
                        if (l_diff != Inf) & (l_diff>-non_equality_tolerance)
                            @constraint(m,av[ind_o+1]>=av[ind_p+1])
                            @constraint(m,av[ind_o+2]>=av[ind_p+2])
                            phi_dep_c[n] = -1
                        end
                        if u_diff<non_equality_tolerance
                            @constraint(m,av[ind_o+1]<=av[ind_p+1])
                            @constraint(m,av[ind_o+2]<=av[ind_p+2])
                            phi_dep_c[n] = 1
                        end
                    end
                end
            end
        end
    end
    phi_dep = reshape(phi_dep_c, size(phi_dep))
    return phi_dep
end

function perturbation_dependencies(m, nn, perturbation, perturbation_size, w, h, k)
    layers_n = layers_number(nn)
    phi_dep = fill(NaN, (1, w, h, k))
    perturbation_init_deps(phi_dep, perturbation, perturbation_size)
    activation_cnt = 1
    println("Encoding dependencies...")
    if reuse_bounds_conf.is_reuse_bounds_and_deps
        for (activation_cnt, phi_dep) in enumerate(reuse_bounds_conf.reusable_deps)
            encode_dependencies(m, layers_n, phi_dep, activation_cnt)
        end
    else
        println("Computing dependencies (once per model and perturbation)...")
        for l in nn.layers
            if occursin("Flatten", string(typeof(l)))
                phi_dep = phi_dep |> l
            elseif occursin("Linear", string(typeof(l))) || occursin("Conv", string(typeof(l)))
                phi_dep_l = dep_propagation(l, phi_dep)
                phi_dep = dep_additional(m, layers_n, l, phi_dep, phi_dep_l, perturbation, perturbation_size, activation_cnt)
            elseif occursin("ReLU", string(typeof(l)))
                phi_dep = encode_dependencies(m, layers_n, phi_dep, activation_cnt)
                push!(reuse_bounds_conf.reusable_deps,phi_dep)
                if all(isnan, phi_dep)
                    break
                end
                activation_cnt += 1
            end
        end
    end
end
