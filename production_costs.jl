using JuMP
using Gurobi

const output_flag = 0

function problem_setup(m, K, seed; total_resource=500.0, u = 1, v = 10)
    rng_gt = MersenneTwister(seed + 200)
    
    b = ones(K) * m * total_resource / K
    # Assume cost is normally distributed around a mean_ik for each good and supplier
    # pre compute costs
    C_mean = u + (v - u) * rand(rng_gt, m, K)
    C = randn(rng_gt, m, n, K)
    for j = 1:n
        C[:, j, :] = C_mean + C[:, j, :] * 0.2
    end
    # r_c_mean = u + (v - u) * rand(rng_gt, m)
    r_c_mean = mean(C_mean, 2) + randn(rng_gt) * 0.2
    temp = r_c_mean
    r_c_mean = zeros(m)
    for i = 1:m
        r_c_mean[i] = temp[i]
    end

    println("Problem:")
    println("b = $b")
    println("Mean Cost Per Item, Supplier")
    for k = 1:K
        println("Supplier $k, mean item cost = $(C_mean[:, k])")
    end
    println("Mean Bid Pricing = $r_c_mean")
    
    rng = MersenneTwister(seed)
    bid_generator = function () return rand(rng, 0:1, m) end
    c_generator = function (j) return C[:, j, :] end
    pi_generator = function (a_k) return transpose(r_c_mean) * a_k + randn(rng) * 0.2 end 
    
    return b, bid_generator, c_generator, pi_generator, rng
end

function offline_lp(env, n, m, K, b, bid_generator::Function, c_generator::Function, pi_generator::Function)
    # collect all data first
    q = zeros(n + 1, K) # resources used
    A = zeros(m, n)
    C = zeros(m, n, K)
    π = zeros(n)
    value = zeros(n + 1)

    for j = 1:n
        A[:, j] = bid_generator()
        C[:, j, :] = c_generator(j)
        π[j] = pi_generator(A[:, j])
    end

    _, prices, X, Y, _ = solve_primal(env, n, m, K, b, A, C, π)

    for j = 1:n
        value[j + 1] = value[j] + X[j] * π[j] - sum(Y[:, j, :] .* C[:, j, :])
        for k = 1:K
            q[j + 1, k] = q[j, k] + sum(Y[:, j, k])
        end
    end

    return prices, X, value, q

end

function online_lp(env, n, h_vector, m, K, b, bid_generator::Function, c_generator::Function, pi_generator::Function;
    greedy=false, print_every=100)
    q = zeros(n + 1, K) # resources used
    A = zeros(m, n)
    C = zeros(m, n, K)
    π = zeros(n)
    prices = zeros(K, n)
    X = zeros(n)
    Y = zeros(m, n, K)
    value = zeros(n + 1)

    # initialize price so that first k bids are not fulfilled
    h = shift!(h_vector)
    h1 = h
    for j = 1:n
        # draw bid
        A[:, j] = bid_generator()
        C[:, j, :] = c_generator(j)
        π[j] = pi_generator(A[:, j])

        # solve ip to get y
        if j >= h1
            feasible, y, obj = ip_sub(env, m, K, b - q[j, :], A[:, j], C[:, j, :], prices[:, j])
            if feasible == true && (π[j] > obj || greedy == true)
                X[j] = 1.0
                Y[:, j, :] = y
            else
                X[j] = 0.0
                Y[:, j, :] = 0
            end
        else
            X[j] = 0.0
            Y[:, j, :] = 0
        end

        for k = 1:K
            q[j + 1, k] = q[j, k] + sum(Y[:, j, k])
        end
        value[j + 1] = value[j] + X[j] * π[j] - sum(Y[:, j, :] .* C[:, j, :])

        # learning step
        if j == h && j < n
            if !isempty(h_vector) h = shift!(h_vector) end
            _, prices[:, j + 1], _, _, _ = solve_primal(env, j, m, K, (b - q[j, :]) * (j / n), A[:, 1:j], C[:, 1:j, :], π[1:j])
        elseif j < n
            prices[:, j + 1] = prices[:, j]
        end
    end
    return prices, X, value, q

end

function solve_primal(env, n, m, K, b, A, C, π)
    model = Model(solver=GurobiSolver(env, OutputFlag = 0))
    @variable(model, 0 <= x[1:n] <= 1)
    @variable(model, y[1:m, 1:n, 1:K] >= 0)

    resource_constraint = []
    for i = 1:m
        for j = 1:n
            push!(resource_constraint, @constraint(model, sum(y[i, j, :]) == A[i, j] * x[j]))
        end
    end

    production_constraint = []
    for k = 1:K
        push!(production_constraint, @constraint(model, sum(y[:, :, k]) <= b[k]))
    end

    @objective(model, Max, sum(x .* π) - sum(C .* y))

    status = solve(model)

    if status == :Optimal
        return getdual(resource_constraint), getdual(production_constraint), getvalue(x), getvalue(y), getobjectivevalue(model)
    else
        print(model)
        error("No production prices found. Status = $status")
    end
end

function solve_dual(env, n, m, K, b, A, C, π)
    model = Model(solver = GurobiSolver(env, OutputFlag=0))
    @variable(model, λ[1:m, 1:n])
    @variable(model, p[1:K] >= 0)
    @variable(model, μ[1:n] >= 0)

    c1 = []
    for j = 1:n
        push!(c1, @constraint(model, μ[j] + sum(A[:, j] .* λ[:, j]) >= π[j]))
    end

    c2 = []
    for i = 1:m
        for j = 1:n
            for k = 1:K
                push!(c2, @constraint(model, λ[i, j] - p[k] <= C[i, j, k]))
            end
        end
    end

    @objective(model, Min, sum(b .* p) + sum(μ))

    status = solve(model)
    
    if status == :Optimal
        return getdual(c1), getdual(c2), getvalue(λ), getvalue(p), getvalue(μ), getobjectivevalue(model)
    else
        print(model)
        error("No optimal solution to dual found. Status = $status")
    end
end

function ip_sub(env, m, K, b, A, C, p)
    model = Model(solver=GurobiSolver(env, OutputFlag=output_flag))
    @variable(model, λ[1:m])
    @variable(model, 0 <= y[1:m, 1:K] <= 1)
    
    for i = 1:m
        @constraint(model, λ[i] == sum(y[i, :] .* (p + C[i, :])))
    end 

    for i = 1:m
        @constraint(model, sum(y[i, :]) == A[i])
    end

    for k = 1:K
        @constraint(model, sum(y[:, k]) <= b[k])
    end

    @objective(model, Min, sum(λ .* A))

    status = solve(model, suppress_warnings=true)

    if status == :Optimal
        return true, getvalue(y), getobjectivevalue(model)
    elseif status == :Infeasible
        return false, zeros(m, K), Inf
    else
        print(model)
        error("Subproblem not optimal or infeasible. Status = $status")
    end
end

function simulation_q6(n, m, K; seed=1234)
    b, bg, cg, pig, rng = problem_setup(m, K, seed)

    env = Gurobi.Env()

    prices = zeros(K, n, 5)
    X = zeros(n, 5)
    value = zeros(n + 1, 5)
    q = zeros(n + 1, K, 5)

    # offline lp
    srand(rng, seed)
    println("Computing offline LP")
    prices_opt, X[:, 1], value[:, 1], q[:, :, 1] = offline_lp(env, n, m, K, b, bg, cg, pig)
    for k = 1:K
        prices[k, :, 1] = prices_opt[k]
    end

    # online lp with k_vector
    h_vector = [50]
    srand(rng, seed)
    println("Computing online LP, h = 50")
    prices[:, :, 2], X[:, 2], value[:, 2], q[:, :, 2]  = online_lp(env, n, h_vector, m, K, b, bg, cg, pig)

    h_vector = [100]
    srand(rng, seed)
    println("Computing online LP, h = 100")
    prices[:, :, 3], X[:, 3], value[:, 3], q[:, :, 3]  = online_lp(env, n, h_vector, m, K, b, bg, cg, pig)

    h_vector = [200]
    srand(rng, seed)
    println("Computing online LP, h = 200")
    prices[:, :, 4], X[:, 4], value[:, 4], q[:, :, 4]  = online_lp(env, n, h_vector, m, K, b, bg, cg, pig)

    # greedy
    h_vector = [50]
    srand(rng, seed)
    println("Computing online greedy LP, h = 50")
    prices[:, :, 5], X[:, 5], value[:, 5], q[:, :, 5]  = online_lp(env, n, h_vector, m, K, b, bg, cg, pig, greedy=true)

    return b, prices, X, value, q
end