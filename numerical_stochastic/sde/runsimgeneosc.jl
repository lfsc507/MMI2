using DifferentialEquations
using LSODA
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
    "--stochtime"
        help = "stochastic simulation endpoint"
        arg_type = Float64
        default = 200.0
end
args = parse_args(parser)

function model_geneosc(du, u, p, t)
    aM, aP, aF, bM, bP, bF, kf, kb, σM, σP, σF, τ = p

    M, P, F, DA, DR = u[1], u[2], u[3], u[4], u[5]

    du[1] = aM * DA - bM * M + u[6]
    du[2] = aP * M - bP * P + u[7]
    du[3] = aF * P - bF * F - kf * F * DA + kb * DR + u[8]
    du[5] = kf * F * DA - kb * DR - bF * DR
    du[4] = -du[5]

    du[6:8] = -u[6:8]/τ
end

function stoch_add(du,u,p,t)
    aM, aP, aF, bM, bP, bF, kf, kb, σM, σP, σF, τ = p
    du[1:5] .= 0
    du[6] = sqrt(2.0*σM^2.0/τ)
    du[7] = sqrt(2.0*σP^2.0/τ)
    du[8] = sqrt(2.0*σF^2.0/τ)
end

function stoch_mul(du,u,p,t)
    aM, aP, aF, bM, bP, bF, kf, kb, σM, σP, σF, τ = p
    du[1:5] .= 0
    du[6] = sqrt(2.0*σM^2.0/τ) * u[1]
    du[7] = sqrt(2.0*σP^2.0/τ) * u[2]
    du[8] = sqrt(2.0*σF^2.0/τ) * u[3]
end

p_kim_add = [
    15.1745, # aM
    1, # aP
    1, # aF
    1, # bM
    1, # bP
    1, # bF
    200, # kf
    50, # kb
    20, # σM
    20, # σP
    2, # σF
    0.1, # τ
]
p_kim_mul = copy(p_kim_add)
p_kim_mul[9:11] .= 1. # σ parameters

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

function sim_geneosc_ic(model, model_stoch, u0, pars; popsize=1000, selectR::String="lo", saveat=0.1)
    selectR == "lo" || selectR == "hi" || error("invalid selectR")
    prob_ode = ODEProblem(model, u0, (0., 200.), pars);
    prob_sde = SDEProblem(model, model_stoch, u0, tspan, pars);
    range = LinRange(0, 500, 20)
    init_conds = [(m, p) for m in range for p in range]
    function prob_func_cells_init(prob, i, repeat)
        m, p = init_conds[i]
        remake(prob, u0=[m; p; u0[3:end]])
    end
    prob_cells_init = EnsembleProblem(prob_ode, prob_func=prob_func_cells_init)
    @time sim_init = solve(prob_cells_init, lsoda(), saveat=saveat, dt=0.01, reltol=1e-6, abstol=1e-6, trajectories=length(init_conds));
    sol_init = sim_init[:, 400:end, :]
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
        remake(prob, u0=u0_s)
    end
    prob_cells = EnsembleProblem(prob_sde, prob_func=prob_func_cells)
    @time sim = solve(prob_cells, STrapezoid(), saveat=saveat, maxiter=1e12, dt=0.0001, trajectories=popsize);
    filter_sols(sim)
end

stochs = Dict("add" => stoch_add, "mul" => stoch_mul)
psets = Dict("add" => p_kim_add, "mul" => p_kim_mul)

stoch = stochs[args["form"]]
u0 = [0., 0., 0., 164.75, 0., 0., 0., 0.]
p_basal = psets[args["form"]]
tspan = (0., args["stochtime"])
noises = eval(Meta.parse(args["noises"]))
signals = eval(Meta.parse(args["signals"]))

all_results = map(noises) do noise
    results = map(signals) do signal
        p = copy(p_basal)
        p[1] = signal
        p[9:11] *= noise
        signal => sim_geneosc_ic(model_geneosc, stoch, u0, p; popsize=args["cells"], selectR=args["initstate"])
    end
    noise => Dict(results)
end
save(args["output"], "results", Dict(all_results))
