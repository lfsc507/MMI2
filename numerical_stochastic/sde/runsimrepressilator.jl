using DifferentialEquations
using LSODA
using Sobol
using JLD2
using ArgParse

parser = ArgParseSettings(allow_ambiguous_opts=false)
@add_arg_table! parser begin
    "output"
        help = "output JLD2"
        required = true
    "form"
        help = "noise form: add or mul"
        required = true
    "target"
        help = "noise target: rna or both"
        required = true
    "initstate"
        help = "initial state: hi or lo"
        required = true
    "signals"
        help = "signal values"
        required = true
    "--cells"
        help = "number of cells to simulate"
        default = 500
        arg_type = Int
    "--noises"
        help = "noise values"
        default = "0.5:0.5:1.5"
end
args = parse_args(parser)

function model_repressilator(du, u, p, t)
    a1, a2, a3, a, n, dR, dP, y, σR, σP, τ = p

    M1, M2, M3, P1, P2, P3 = u[1], u[2], u[3], u[4], u[5], u[6]

    du[1] = a + a1 / (1 + P3^n) - dR * M1 + u[7]
    du[2] = a + a2 / (1 + P1^n) - dR * M2 + u[8]
    du[3] = a + a3 / (1 + P2^n) - dR * M3 + u[9]
    
    du[4] = y * M1 - dP * P1 + u[10]
    du[5] = y * M2 - dP * P2 + u[11]
    du[6] = y * M3 - dP * P3 + u[12]

    du[7:9] = - u[7:9]/τ
    du[10:12] = - u[10:12]/τ
end

function stoch_repressilator_add(du,u,p,t)
    a1, a2, a3, a, n, dR, dP, y, σR, σP, τ = p
    du[1:6] .= 0
    du[7:9] .= sqrt((2.0*σR^2.0/τ))
    du[10:12] .= sqrt((2.0*σP^2.0/τ))
end

function stoch_repressilator_mul(du,u,p,t)
    a1, a2, a3, a, n, dR, dP, y, σR, σP, τ = p
    du[1:6] .= 0
    du[7:9] = sqrt(2.0*σR^2.0/τ) .* u[1:3]
    du[10:12] = sqrt(2.0*σP^2.0/τ) .* u[4:6]
end

p_cycle = [
    7, 5, 5, # a1, a2, a3
    0.01, # a
    2, # n
    1, 1, # dR, dP
    1, # y
    1, 1, # σR, σP
    0.1 # τ
];

function filter_sols(sols)
    nvars, ntps, nsols = size(sols)
    good_sols = []
    for i in 1:nsols
        if isassigned(sols[i][1,:], ntps) == true
            push!(good_sols, i)
        end
    end
    println("Obtained $(length(good_sols))/$nsols converged solutions.")
    sols[:, :, good_sols]
end

function sim_pop_ic(model, model_stoch, u0, pars; popsize=1000, selectR::String="lo", saveat=1.0)
    selectR == "lo" || selectR == "hi" || error("invalid selectR")
    prob_ode = ODEProblem(model, u0, (0., 5000.), pars);
    prob_sde = SDEProblem(model, model_stoch, u0, tspan, pars);
    seq = SobolSeq(3)
    init_rnas = [next!(seq) .* 5 .+ [0.01, 0, 0] for _ in 1:100]
    function prob_func_cells_init(prob,i,repeat)
        init_rna = init_rnas[i]
        remake(prob,u0=[init_rna; init_rna; u0[7:end]])
    end
    prob_cells_init = EnsembleProblem(prob_ode, prob_func=prob_func_cells_init)
    @time sim_init = solve(prob_cells_init, lsoda(), saveat=saveat, dt=0.01, reltol=1e-6, abstol=1e-6, trajectories=length(init_rnas));
    sol_init = sim_init[:, 4000:end, :] # The selection of state is sensitive to the start index
    if selectR == "lo"
        sol_init[isnan.(sol_init)] .= Inf
        idsel = argmin(sol_init[1, :, :])
    else
        sol_init[isnan.(sol_init)] .= -Inf
        idsel = argmax(sol_init[1, :, :])
    end
    u0_s = sol_init[:, idsel[1], idsel[2]]
    println("Starting stochastic simulation with init cond ", u0_s)
    function prob_func_cells(prob, i, repeat)
        remake(prob,u0=u0_s)
    end
    prob_cells = EnsembleProblem(prob_sde, prob_func=prob_func_cells)
    @time sim = solve(prob_cells, STrapezoid(), saveat=saveat, maxiter=1e12, dt=0.0001, trajectories=popsize);
    filter_sols(sim)
end

stochs = Dict("add" => stoch_repressilator_add, "mul" => stoch_repressilator_mul)

stoch = stochs[args["form"]]
tspan = (0., 1000.)
u0 = [1.; zeros(11)]
noises = eval(Meta.parse(args["noises"]))
signals = eval(Meta.parse(args["signals"]))

all_results = map(noises) do noise
    results = map(signals) do signal
        p = copy(p_cycle)
        p[1] = signal
        p[9] = noise
        p[10] = args["target"] == "both" ? noise : 0.
        result = sim_pop_ic(model_repressilator, stoch, u0, p; popsize=args["cells"], selectR=args["initstate"])
        signal => result[1:6, :, :]
    end
    noise => Dict(results)
end
save(args["output"], "results", Dict(all_results))
