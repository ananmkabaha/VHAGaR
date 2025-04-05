ENV["PYTHON"]="/usr/bin/python3.8"

using Gurobi
using PyCall
using PyPlot
using Gurobi
using Images
using Printf
using Dates
using Base.Cartesian
using JuMP
using MathOptInterface
using Memento
using DocStringExtensions
using ProgressMeter
using ArgParse
using Serialization

np = pyimport("numpy")

include("utils/MIPVerify.jl/src/MIPVerify.jl")
const dependencies_path = joinpath(@__DIR__, "utils/MIPVerify.jl/", "deps")
@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3
const DEFAULT_TIGHTENING_ALGORITHM = mip

include("utils/MIPVerify.jl/src/vendor/ConditionalJuMP.jl")
include("utils/MIPVerify.jl/src/net_components.jl")
include("utils/perturbation_dependencies.jl")
include("utils/MIPVerify.jl/src/logging.jl")
include("utils/MIPVerify.jl/src/models.jl")
include("utils/MIPVerify.jl/src/utils.jl")
include("utils/perturbation_models.jl")
include("utils/help_functions.jl")
include("utils/hyper_attack.jl")
include("utils/datasets.jl")
include("utils/models.jl")
include("utils/mip.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "mnist, fmnist, or cifar10"
        arg_type = String
        required = false
        default = "mnist"
        "--model_name", "-n"
        help = "FC0, FC1, CNN0, CNN1, or CNN2"
        arg_type = String
        required = false
        default = "FC0"
        "--model_path", "-m"
        help = "model name"
        arg_type = String
        required = false
        default = "./models/mnist_FC0/model.p"
        "--perturbation", "-p"
        help = "perturbation type: occ, patch, brightness, linf, contrast, translation, rotation, or max"
        arg_type = String
        required = false
        default = "brightness"
        "--perturbation_size", "-s"
        help = "occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle"
        arg_type = String
        required = false
        default = "0.1"
        "--ctag", "-c"
        help = "ctag, source class"
        arg_type = Int
        required = false
        default = 1
        "--timeout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 10800
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2,3,4,5,6,7,8,9,10"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "./results/"
        "--is_debug"
        help = "Output debug files"
        arg_type = Bool
        required = false
        default = false
        "--is_reuse_bounds_and_deps"
        help = "Output debug files"
        arg_type = Bool
        required = false
        default = true
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = args["model_name"]
    model_path = args["model_path"]
    perturbation = args["perturbation"]
    perturbation_size = parse_perturbations(perturbation, args["perturbation_size"])
    c_tag = args["ctag"]
    c_targets = parse_numbers_to_Int64(args["ct"])
    results_path = args["output_dir"]
    is_debug = args["is_debug"]
    is_reuse_bounds_and_deps = args["is_reuse_bounds_and_deps"]
    timeout = args["timeout"]

    w, h, k, c = get_dataset_params( dataset )
    nn = get_nn(model_path, model_name, w, h, k, c, dataset)
    token_signature = string(now().instant.periods.value)
    reuse_bounds_and_deps(is_reuse_bounds_and_deps, reuse_bounds_conf, dataset, model_name, perturbation, perturbation_size)
    for c_target in c_targets
        if c_tag == c_target && perturbation != "max"
            println("Avoid running c_tag=c_target")
            continue
        end
        suboptimal_solution, suboptimal_time =  hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size)
        optimizer = Gurobi.Optimizer
        d = Dict()
        d[:TargetIndex] = get_target_indexes(c_target, c)
        d[:SourceIndex] = get_target_indexes(c_tag, c)
        d[:suboptimal_solution] = suboptimal_solution
        d[:suboptimal_time] = suboptimal_time
        mip_reset()
        bounds_time = @elapsed begin
            merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn, zeros(Float64, 1, w, h, k), optimizer,
             get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
        end
        d[:bounds_time] = bounds_time
        m = d[:Model]
        hyper_attack_hints(m, token_signature, c_tag, c_target, perturbation)
        perturbation_dependencies(m, nn, perturbation, perturbation_size, w, h, k)
        mip_set_delta_property(m, perturbation, d)
        set_optimizer(m, optimizer)
        mip_set_attr(m, perturbation, d, timeout)
        MOI.set(m, Gurobi.CallbackFunction(), my_callback)
        optimize!(m)
        mip_log(m, d)
        mip_reuse_bounds()
        results.str = update_results_str(results.str, c_tag, c_target, d)
        save_results(is_debug, results_path, dataset, model_name, perturbation, perturbation_size, results.str, d, nn, c_tag-1, c_target-1, w, h, k)
        if perturbation == "max"
            break
        end
        log_reuse_bounds_and_deps(is_reuse_bounds_and_deps, reuse_bounds_conf, dataset, model_name, perturbation, perturbation_size)
    end
    println("Results:"*"\n"*results.str)
end

main()