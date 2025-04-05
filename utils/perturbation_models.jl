function get_model(w_, h_, k_,
    perturbation,
    perturbation_size,
    nn::NeuralNet,
    input::Array{<:Real},
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
)::Dict{Symbol,Any}
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    if perturbation == "contrast"
        set_optimizer_attribute(m, "NonConvex", 2)
    end
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)
    d_common = Dict(
        :Model => m,
        :TighteningApproach => string(tightening_algorithm),
    )
    println("Encoding the two copies...")
    if reuse_bounds_conf.is_reuse_bounds_and_deps == false
        println("Computing the bounds of the two copies (once per model and perturbation)...")
    end

    if perturbation == "brightness"
        return merge(d_common, get_perturbation_specific_keys_brightness(perturbation_size,nn, input, m))
    elseif perturbation == "linf"
        return merge(d_common, get_perturbation_specific_keys_linf(perturbation_size,nn, input, m))
    elseif perturbation == "max"
        return merge(d_common, get_perturbation_specific_keys_max(perturbation_size,nn, input, m))
    elseif perturbation == "contrast"
        return merge(d_common, get_perturbation_specific_keys_contrast(perturbation_size,nn, input, m))
    elseif perturbation == "occ"
        return merge(d_common, get_perturbation_specific_keys_occ(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "patch"
        return merge(d_common, get_perturbation_specific_keys_patch(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "patchM"
        return merge(d_common, get_perturbation_specific_keys_patchM(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "occM"
        return merge(d_common, get_perturbation_specific_keys_occM(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "translation"
        return merge(d_common, get_perturbation_specific_keys_translation(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "rotation"
        return merge(d_common, get_perturbation_specific_keys_rotate(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "filterv"
        return merge(d_common, get_perturbation_specific_keys_filter_v(perturbation_size,nn, input, m))
     elseif perturbation == "Privacy"
        return merge(d_common, get_perturbation_specific_keys_privacy(w_, h_, k_, perturbation_size, nn, nn_second, input, m))
    else
        return merge(d_common, get_perturbation_specific_keys(perturbation_size,nn, input, m))
    end
end


function get_perturbation_specific_keys_linf(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range,)
    v_in = map( i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    @constraint(m, v_x0 .== v_in + v_e)
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_privacy(w_, h_, k_, perturbation_size, nn::NeuralNet, nn_hyper::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_in_output = v_in |> nn
    v_output = v_in |> nn_hyper
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_brightness(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = @variable(m, lower_bound = 0, upper_bound = p_size)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1),input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1+p_size), input_range,)
    @constraint(m, v_x0 .== v_in .+ v_e)
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_max(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_output = v_in |> nn
    return Dict(:v_in_p => v_in, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_output)
end

#contrast
function get_perturbation_specific_keys_contrast(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = @variable(m, lower_bound = 1.0, upper_bound = 1+p_size)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1+p_size), input_range,)
    @constraint(m, v_x0 .== v_e*v_in)
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_occ(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    l = []
    ind1 = Int(perturbation_size[1])
    ind2 = Int(perturbation_size[2])
    w = w_
    h = h_
    k = k_
    res_ = w*h
    ind = ind1 + (ind2-1)*w
    for i_ in 0:Int(perturbation_size[3])-1
        for j_ in 0:Int(perturbation_size[3])-1
            append!(l, ind+j_+w*i_)
            if k == 3
                append!(l, res_+ind+j_+w*i_)
                append!(l, 2*res_+ind+j_+w*i_)
            end
        end
    end
    res = []
    for tt in 1:Int(w_*h_*k_)
        if tt in l
            continue
        end
        append!(res, tt)
    end
    @constraint(m, c1[i=l],v_x0[i] == 0.0)
    @constraint(m, c2[i=res],v_x0[i] == v_in[i])
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_patch(w_, h_, k_, perturbation_size, nn::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    l = []
    eps  = perturbation_size[1]
    ind1 = Int(perturbation_size[2])
    ind2 = Int(perturbation_size[3])
    w = w_
    h = h_
    k = k_
    res_ = w*h

    ind = ind1 + (ind2-1)*w
    for i_ in 0:Int(perturbation_size[4])-1
        for j_ in 0:Int(perturbation_size[4])-1
            append!(l, ind+j_+w*i_)
            if k == 3
                append!(l, res_+ind+j_+w*i_)
                append!(l, 2*res_+ind+j_+w*i_)
            end
        end
    end
    res = []
    for tt in 1:Int(w_*h_*k_)
        if tt in l
            continue
        end
        append!(res, tt)
    end
    @constraint(m, c0[i=l],v_x0[i] <= v_in[i]+eps)
    @constraint(m, c1[i=l],v_x0[i] >= v_in[i]-eps)
    @constraint(m, c2[i=res],v_x0[i] == v_in[i])

    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_patchM(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
   #TPD
end

function get_perturbation_specific_keys_occM(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    #TPD
end

function get_perturbation_specific_keys_translation(w_, h_, k_,perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    t_down = perturbation_size[1]
    t_right = perturbation_size[2]
    k = Int(k_)
    w = Int(w_)
    h = Int(h_)
    res = Int(w*h)
    m_ind = Int(t_down)
    n_ind = Int(t_right)
    for i2 = 1:w-t_right
        for i1 = 1:h-t_down
            i = Int(i1 + h *(i2-1))
            @constraint(m,v_x0[Int(i+m_ind+w*n_ind)] == v_in[i])
            if k == 3
                i = Int(res+i1 + h *(i2-1))
                @constraint(m,v_x0[Int(i+m_ind+w*n_ind)] == v_in[i])
                i = Int(2*res+i1 + h *(i2-1))
                @constraint(m,v_x0[Int(i+m_ind+w*n_ind)] == v_in[i])
            end
        end
    end

    for j = 1:t_down
        @constraint(m,[i=j:w:res],v_x0[Int(i)] == 0)
        if k == 3
            @constraint(m,[i=j+res:w:2*res],v_x0[Int(i)] == 0)
            @constraint(m,[i=j+2*res:w:3*res],v_x0[Int(i)] == 0)
        end
    end
    for j = 1:t_right
        @constraint(m,[i=1+w*(j-1):1:w*j],v_x0[Int(i)] == 0)
        if k == 3
            @constraint(m,[i=res+1+w*(j-1):1:res+w*j],v_x0[Int(i)] == 0)
            @constraint(m,[i=res*2+1+w*(j-1):1:res*2+w*j],v_x0[Int(i)] == 0)
        end
    end
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_filter_v(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)

    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    for j=1:27
        @constraint(m,[i=28*j+1:28*(j+1)-1],v_x0[i]==0.01*v_in[i-1]+0.99*v_in[i]+0.01*v_in[i+1])
    end
    @constraint(m,[i=28:28:784],v_x0[i]== 0.1*v_in[i-1]+0.8*v_in[i])
    @constraint(m,[i=1:28:756],v_x0[i]== 0.8*v_in[i]+0.1*v_in[i+1])
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end


function get_perturbation_specific_keys_rotate(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    angle = perturbation_size[1]
    k = k_
    height = h_
    width = w_
    res_ = h_*w_
    center = [width/2, height/2]
    l = []
    for i = 1:height
        for j = 1:width
            j_c = j - center[1]
            i_c = i - center[2]
            j_r = (j_c*cos(angle*pi/180) - i_c*sin(angle*pi/180) + center[1])
            i_r = (j_c*sin(angle*pi/180) + i_c*cos(angle*pi/180) + center[2])
            if floor(Int,j_r) >= 1 && ceil(Int,j_r) <= width && floor(Int,i_r) >= 1 && ceil(Int,i_r) <= height
                di = i_r-floor(i_r)
                dj = j_r-floor(j_r)
                @constraint(m,v_x0[i+(j-1)*height] == (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height]+
                (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height]+(1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height]+
                (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height])
                append!(l, i+(j-1)*height)
                if k==3
                    @constraint(m,v_x0[i+(j-1)*height+res_] == (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height+res_]+
                    (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height+res_]+(1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height+res_]+
                    (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height+res_])
                    append!(l, i+(j-1)*height+res_)
                    @constraint(m,v_x0[i+(j-1)*height+2*res_] == (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height+2*res_]+
                    (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height+2*res_]+(1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height+2*res_]+
                    (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height+2*res_])
                    append!(l, i+(j-1)*height+2*res_)
                end
            end
        end
    end
    for tt in 1:res_
        if tt in l
            continue
        end
        @constraint(m,v_x0[tt] == 0)
    end
    v_in_output = v_in |> nn
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end


