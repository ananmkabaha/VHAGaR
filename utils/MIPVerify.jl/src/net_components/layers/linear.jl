export Linear

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`Linear`.

## Fields:
$(FIELDS)
"""
mutable struct Linear{T<:Real,U<:Real} <: Layer
    matrix::Array{T,2}
    bias::Array{U,1}

    function Linear{T,U}(matrix::Array{T,2}, bias::Array{U,1}) where {T<:Real,U<:Real}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias)
    end

end

function Linear(matrix::Array{T,2}, bias::Array{U,1}) where {T<:Real,U<:Real}
    Linear{T,U}(matrix, bias)
end

function Base.show(io::IO, p::Linear)
    input_size = size(p.matrix)[1]
    output_size = size(p.matrix)[2]
    print(io, "Linear($input_size -> $output_size)")
end

function check_size(params::Linear, sizes::NTuple{2,Int})::Nothing
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end],))
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`.
"""
function matmul(x::Array{<:Real,1}, params::Linear)
    return transpose(params.matrix) * x .+ params.bias
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`. We write the computation out by hand when working with `JuMPLinearType`
so that we are able to simplify the output as the computation is carried out.
"""
function matmul(x::Array{T,1}, params::Linear{U,V}) where {T<:JuMPLinearType,U<:Real,V<:Real}
    Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
    (matrix_height, matrix_width) = size(params.matrix)
    (input_height,) = size(x)
    @assert(
        matrix_height == input_height,
        "Number of values in input, $input_height, does not match number of values, $matrix_height that Linear operates on."
    )

    return transpose(params.matrix) * x .+ params.bias
end

(p::Linear)(x::Array{<:JuMPReal}) =
    "Linear() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |>
    ArgumentError |>
    throw

(p::Linear)(x::Array{<:JuMPReal,1}) = matmul(x, p)
