#!/usr/bin/julia

using SparseArrays;
using IterTools;
using LinearAlgebra;
using Mosek;
using Combinatorics;

function e(dim::Int64, index::Int64)
    return SparseVector(dim, [index], [1])
end

function sym_kron(vec_list::Union{Vector{SparseVector{Int64, Int64}}, Vector{SparseVector{Float64, Int64}}}, sym_dim::Int, sym_dict::Dict{Any, Any})
    result_index = Int64[]
    result_value = typeof(vec_list[1][1])[]
    for index_val_list in product([zip(findnz(vec)[1],findnz(vec)[2]) for vec in vec_list]...)
        push!(result_index, sym_dict[sort!([i[1] for i in index_val_list])])
        push!(result_value, prod([i[2] for i in index_val_list]))
    end
    result = sparsevec(result_index, result_value, sym_dim)
    return result
end

function swap(dim1::Int64, dim2::Int64, dim_between::Int64 = 1)
    mat = spzeros(dim1 * dim2 * dim_between, dim1 * dim2 * dim_between)

    for i in 1:dim1, j in 1:dim2
        mat += kron([e(dim2,j) * e(dim1,i)', sparse(1*I, dim_between, dim_between), e(dim1,i) * e(dim2,j)']...)
    end

    return mat
end

function get_pos_cons(n::Int, labelDict_43::Dict{Any, Any}, sym_dim43::Int, labelDict_34::Dict{Any, Any}, sym_dim34::Int)
    # CS43 positive functionals
    CS43_1 = e(10,1)
    CS43_simple = [e(10,i) for i in 2:10]
    
    CS43_sum = SparseVector{Int64, Int64}[]
    for i in 0:2
        vec = ones(1)
        append!(vec, zeros(3*i))
        append!(vec, -ones(3))
        append!(vec, zeros(3*(2-i)))
        @assert size(vec) == size(CS43_1)
        push!(CS43_sum, sparse(vec))
    end

    CS43_all = union(CS43_simple, CS43_sum)

    # CS 34 positive functionals
    CS34_1 = e(9, 1)
    CS34_simple = [e(9, i) for i in 2:9]
    
    CS34_sum = SparseVector{Int64, Int64}[]
    for i in 0:3
        vec = ones(1)
        append!(vec, zeros(2*i))
        append!(vec, -ones(2))
        append!(vec, zeros(2*(3-i)))
        @assert size(vec) == size(CS34_1)
        push!(CS34_sum, sparse(vec))
    end
    
    CS34_all = union(CS34_simple, CS34_sum)

    pos_cons = SparseVector{Int64, Int64}[]
    pos_cons_threads = [SparseVector{Int64, Int64}[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for f_sum in CS43_sum
        for f_all_list in multiset_combinations(CS43_all, (n-1) * ones(Int64, length(CS43_all)), n-1)
            for g_all_list in multiset_combinations(CS34_all, n * ones(Int64, length(CS43_all)), n)
                f_list = SparseVector{Int64, Int64}[]
                push!(f_list, f_sum)
                append!(f_list, f_all_list)
                push!(pos_cons_threads[Threads.threadid()], kron(sym_kron(f_list, sym_dim_43, labelDict_43), sym_kron(g_all_list, sym_dim_34, labelDict_34)))
            end
        end
    end
    
    Threads.@threads for f_simple_list in collect(multiset_combinations(CS43_simple, n * ones(Int64, length(CS43_simple)), n))
        for g_sum in CS34_sum
            for g_all_list in multiset_combinations(CS34_all, (n-1) * ones(Int64, length(CS34_all)), n-1)
                g_list = SparseVector{Int64, Int64}[]
                push!(g_list, g_sum)
                append!(g_list, g_all_list)
                push!(pos_cons_threads[Threads.threadid()], kron(sym_kron(f_simple_list, sym_dim_43, labelDict_43), sym_kron(g_list, sym_dim_34, labelDict_34)))
            end
        end
    end
    
    append!(pos_cons, pos_cons_threads...)

    return pos_cons
end

function get_shuffle(n::Int)
    shuffle = sparse(1*I, 90^n, 90^n)

    for k in 1:(n ÷ 2)
        index_B = 2*k - 1 + (n % 2)
        shuffle *= kron(sparse(1*I, 10^(2*k-1), 10^(2*k-1)), swap(9, 10, 10^(n-2*k) * 9^(index_B - 1)), sparse(1*I, 9^(n - index_B), 9^(n - index_B)))
    end

    return shuffle
end

function get_F_cons(n::Int, M::Matrix{Float64}, sym_proj_shuffle::SparseMatrixCSC{Float64, Int64})
    # CS43 positive functionals
    CS43_1 = e(10,1)
    CS43_m = [SparseVector{Int64, Int64}(spzeros(10)) for i in 1:4, j in 1:3]
    for i in 1:4, j in 1:3
        if i != 4
            CS43_m[i, j] = e(10,1 + 3*(j-1) + i)
        else
            CS43_m[i, j] = e(10,1)
            for k in 1:3
                CS43_m[i, j] -= e(10,1 + 3*(j-1) + k)
            end
        end
    end
    #CS34 positive functionals
    CS34_1 = e(9, 1)
    CS34_m = [SparseVector{Int64, Int64}(spzeros(9)) for i in 1:3, j in 1:4]
    for i in 1:3, j in 1:4
        if i != 3
            CS34_m[i, j] = e(9,1 + 2*(j-1) + i)
        else
            CS34_m[i, j] = e(9,1)
            for k in 1:2
                CS34_m[i, j] -= e(9,1 + 2*(j-1) + k)
            end
        end
    end
    # tensor product of traces
    CS4334_1 = kron(CS43_1, CS34_1)

    F = [SparseVector{Float64, Int64}(spzeros(90)) for i in 1:4, j in 1:4]
    
    for i in 1:4, j in 1:4
        F[i, j] = - M[i,j] * CS4334_1
        for k in 1:3
            F[i, j] += kron(CS43_m[i,k], CS34_m[k,j])
        end
        dropzeros!(F[i,j])
    end

    shuffle = get_shuffle(n)

    F_cons = SparseVector{Float64, Int64}[]
    if n == 1
        for i in 1:4, j in 1:4
            push!(F_cons, F[i,j])
        end
    else
        F_cons_threads = [SparseVector{Float64, Int64}[] for i in 1:Threads.nthreads()]
        for p in 1:n
            Threads.@threads for index_list in collect(multiset_permutations(1:4, 2*p * ones(Int64, 4), 2*p))
                push!(F_cons_threads[Threads.threadid()], sym_proj_shuffle * kron(append!([F[index_list[2*k - 1], index_list[2*k]] for k in 1:p], [CS4334_1 for i in 1:n-p])...))
            end
        end
        append!(F_cons, F_cons_threads...)
    end

    return F_cons
end

function run_LP(n::Int, pos_cons, F_cons)
    maketask() do task
        putstreamfunc(task,MSK_STREAM_LOG,msg -> print(msg))
    
        putintparam(task, MSK_IPAR_NUM_THREADS, Threads.nthreads()) # use all cores
        putintparam(task, MSK_IPAR_INTPNT_BASIS, 0) # do not find basis
    
        num_vars = 90^n
    
        appendvars(task, num_vars)
    
        println("Adding variables...")
    
        putvarname(task, 1, "λ_1")
        putvarbound(task, 1, MSK_BK_FX, 1.0, 1.0)
    
        for i in 2:num_vars
            putvarname(task, i, "λ_$(i)")
            putvarbound(task, i, MSK_BK_RA, 0.0, 1.0)
        end
    
        appendcons(task, length(pos_cons) + length(F_cons))
    
        println("Adding positivity constraints...")
    
        for i in 1:length(pos_cons)
            index = i
            a_index, a_row = findnz(pos_cons[i])
            putarow(task, index, a_index, a_row)
            putconbound(task, index,  MSK_BK_RA, 0.0, 1.0)
        end
    
        println("Adding F constraints...")
    
        for i in 1:length(F_cons)
            index = length(pos_cons) + i
            a_index, a_row = findnz(F_cons[i])
            putarow(task, index, a_index, a_row)
            putconbound(task, index,  MSK_BK_FX, 0.0, 0.0)
        end
    
        putclist(task, [1], [0.0])
        putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE)
    
        #println()
        #println(task)
    
        println("Optimizing...")
        println()
    
        optimize(task)
    
        println("Done.")
        println()
    
        # var = getxx(task, MSK_SOL_ITR)
    
        println(getprosta(task, MSK_SOL_ITR))
        println(getsolsta(task,MSK_SOL_ITR))
        return getprosta(task, MSK_SOL_ITR)
    
    end
    
    
end

function get_M(a::Real,b::Real)
    return 0.25 * [1-a 1+a 1-b 1+b;
                   1+a 1-a 1-b 1+b;
                   1+a 1-a 1+b 1-b;
                   1-a 1+a 1+b 1-b];
end

function bisection(alpha::Real, r0::Real, r1::Real, eps::Real, MaxIter::Int, posConstr)
    
    # Writes output to file so that it can be stored and analysed
    io = open("out_file.txt", "a")
    redirect_stdout(io) do
        println("r0 is ", r0, " and r1 is ", r1)

        if r1 - r0 >= eps && MaxIter >= 0
            rnew = (r0 + r1)/2
            a = 1 - rnew*sin(alpha)
            b = 1 - rnew*cos(alpha)

            solution = run_LP(3, posConstr, get_F_cons(3, get_M(a,b), sym_proj_shuffle))
        
            if solution == Mosek.MSK_PRO_STA_PRIM_INFEAS
                r0 = rnew
            else #This also catches "UNKNOWN" status, which is intended: we want to detect infeasibility with certainty
                r1 = rnew
            end
        
        solution = 0 #freeing up the memory
        end
    end
    close(io)
    
    
    if r1 - r0 >= eps && MaxIter >= 0
        r0, r1 = bisection(alpha, r0, r1, eps, MaxIter-1, posConstr)
    end
    
    return r0, r1
    
end
