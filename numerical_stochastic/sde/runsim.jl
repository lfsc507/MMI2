using DifferentialEquations
using LSODA
using JLD2
using ArgParse

parser = ArgParseSettings(allow_ambiguous_opts=false)
@add_arg_table! parser begin
    "output"
        help = "output JLD2"
        required = true
    "pset"
        help = "parameter set: snic, sn, hopf, sl"
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

function model_mmi2_add(du, u, p, t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, σr, σR, τ = p

    sig = sR

    r, R, c1A, c1B, c2 = u[1], u[2], u[3], u[4], u[5]

    du[1] = dr = sr - kr * r + koff * (c1A+c1B) - K1 * koff * 2R * r +
    koff * 2c2 - K2 * koff * (c1A+c1B) * r +
    (c1A+c1B) * kR * a1 + 2c2 * kR * a2 + u[6]

    du[2] = dR = sig + sR0 - kR * R + koff * (c1A+c1B) - K1 * koff * 2R * r +
    (c1A+c1B) * kr * b1 + u[7]

    du[3] = dc1A = K1 * koff * R * r - koff * c1A +
    koff * c2 - K2 * koff * c1A * r +
    c2 * 1kr * b2 - c1A * kR * a1 - c1A * kr * b1

    du[4] = dc1B = K1 * koff * R * r - koff * c1B +
    koff * c2 - K2 * koff * c1B * r +
    c2 * 1kr * b2 - c1B * kR * a1 - c1B * kr * b1

    du[5] = dc2 = K2 * koff * (c1A+c1B) * r - koff * 2c2 -
    c2 * 2kr * b2 - c2 * kR * a2

    du[6] = - u[6]/τ
    du[7] = - u[7]/τ
end

function stoch_add(du,u,p,t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, σr, σR, τ = p
    du[1] = 0 
    du[2] = 0
    du[3] = 0
    du[4] = 0
    du[5] = 0
    du[6] = sqrt((2.0*σr^2.0/τ))
    du[7] = sqrt((2.0*σR^2.0/τ))
end

function model_mmi2_mul(du, u, p, t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, σr, σR, τ = p
    
    sig = sR

    r, R, c1A, c1B, c2 = u[1], u[2], u[3], u[4], u[5]

    du[1] = dr = sr - kr * r + koff * (c1A+c1B) - K1 * koff * 2R * r +
    koff * 2c2 - K2 * koff * (c1A+c1B) * r +
    (c1A+c1B) * kR * a1 + 2c2 * kR * a2 + u[6]

    du[2] = dR = sig + sR0 - kR * R + koff * (c1A+c1B) - K1 * koff * 2R * r +
    (c1A+c1B) * kr * b1 + u[7]

    du[3] = dc1A = K1 * koff * R * r - koff * c1A +
    koff * c2 - K2 * koff * c1A * r +
    c2 * 1kr * b2 - c1A * kR * a1 - c1A * kr * b1 + u[8]

    du[4] = dc1B = K1 * koff * R * r - koff * c1B +
    koff * c2 - K2 * koff * c1B * r +
    c2 * 1kr * b2 - c1B * kR * a1 - c1B * kr * b1 + u[9]

    du[5] = dc2 = K2 * koff * (c1A+c1B) * r - koff * 2c2 -
    c2 * 2kr * b2 - c2 * kR * a2 + u[10]

    du[6] = - u[6]/τ
    du[7] = - u[7]/τ
    du[8] = - u[8]/τ
    du[9] = - u[9]/τ
    du[10] = - u[10]/τ
end


function stoch_mul(du,u,p,t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, σr, σR, τ = p
    du[1] = 0 
    du[2] = 0
    du[3] = 0
    du[4] = 0
    du[5] = 0
    du[6] = sqrt((2.0*σR^2.0/τ))*u[1] 
    du[7] = sqrt((2.0*σR^2.0/τ))*u[2]
    du[8] = sqrt((2.0*σR^2.0/τ))*u[3] 
    du[9] = sqrt((2.0*σR^2.0/τ))*u[4]
    du[10] = sqrt((2.0*σR^2.0/τ))*u[5]
end

p_snic = [ # SNIC, like Figure 5B
    0.3, #1 sR
    1.0, #2 a1
    12.0, #3 a2
    1.0, #4 b1
    4.0, #5 b2
    1.0, #6 kR
    0.25, #7 kr
    1000.0, #8 K1
    1000.0, #9 K2
    100.0, #10 koff
    1.0, #11 sr
    5.7, #12 sR0    total 6
    0.00, #13 σr    replaced
    0.4, #14 σR     replaced
    0.1 #15 τ
   ];

p_hopf = [ # Diverging Hopf, like Figure 3B
   0.5, #1 sR
   1.0, #2 a1
   12.0, #3 a2
   1.0, #4 b1
   7.0, #5 b2
   1.0, #6 kR
   0.25, #7 kr
   1000.0, #8 K1
   1000.0, #9 K2
   100.0, #10 koff
   1.0, #11 sr
   3.1, #12 sR0    total 3.6
   0.00, #13 σr    replaced
   0.4, #14 σR     replaced
   0.1 #15 τ
  ];

p_sn = [ # SN, like Figure 3F
    3.0, #1 sR
    1.0, #2 a1
    4.0, #3 a2
    0.5, #4 b1
    0.1, #5 b2
    1.0, #6 kR
    2.0, #7 kr
    1000.0, #8 K1
    1000.0, #9 K2
    100.0, #10 koff
    1.0, #11 sr
    0.0, #12 sR0    total 3
    0.00, #13 σr    replaced
    0.4, #14 σR     replaced
    0.1 #15 τ
   ];

p_sl = [ # Saddle-loop, like Figure 5G
   1.0, #1 sR
   1.0, #2 a1
   12.0, #3 a2
   1.0, #4 b1
   3.0, #5 b2
   1.0, #6 kR
   0.25, #7 kr
   1000.0, #8 K1
   1000.0, #9 K2
   100.0, #10 koff
   1.0, #11 sr
   6.9, #12 sR0    total 7.9
   0.00, #13 σr    replaced
   0.4, #14 σR     replaced
   0.1 #15 τ
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

function sim_pop_ic(model, model_stoch, u0, pars; popsize=1000, selectR::String="lo", saveat=0.1)
    selectR == "lo" || selectR == "hi" || error("invalid selectR")
    prob_ode = ODEProblem(model, u0, (0., 200.), pars);
    prob_sde = SDEProblem(model, model_stoch, u0, tspan, pars);
    r0s = range(0.001, 10.0, length=20)
    R0s = range(0.001, 10.0, length=20)
    function prob_func_cells_init(prob,i,repeat)
        m, n = i ÷ length(r0s), i % length(r0s)
        if n != 0
            r0, R0 = r0s[m+1],  R0s[n]
        else
            r0, R0 = r0s[m], R0s[end]
        end
        remake(prob, u0=[r0; R0; u0[3:end]])
    end
    prob_cells_init = EnsembleProblem(prob_ode, prob_func=prob_func_cells_init)
    @time sim_init = solve(prob_cells_init, lsoda(), saveat=saveat, dt=0.01, 
        reltol=1e-6, abstol=1e-6, trajectories=length(r0s)*length(R0s));
    sol_init = sim_init[:, 400:end, :] # The selection of state is sensitive to the start index
    sol_reduced = dropdims(sum(sol_init[2:5, :, :]; dims=1), dims=1)
    if selectR == "lo"
        sol_reduced[isnan.(sol_reduced)] .= Inf
        idsel = argmin(sol_reduced)
    else
        sol_reduced[isnan.(sol_reduced)] .= -Inf
        idsel = argmax(sol_reduced)
    end
    u0_s = sol_init[:, idsel[1], idsel[2]]
    println("Starting stochastic simulation with init cond ", u0_s)
    function prob_func_cells(prob,i,repeat)
        remake(prob,u0=u0_s)
    end
    prob_cells = EnsembleProblem(prob_sde, prob_func=prob_func_cells)
    @time sim = solve(prob_cells, STrapezoid(), saveat=saveat, maxiter=1e12, dt=0.0001, trajectories=popsize);
    filter_sols(sim)
end

u0s = Dict("add" => [0.8, 0.1, 0., 0., 0., 0., 0.], "mul" => [0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0.])
models = Dict("add" => model_mmi2_add, "mul" => model_mmi2_mul)
stochs = Dict("add" => stoch_add, "mul" => stoch_mul)
psets = Dict("snic" => p_snic, "sn" => p_sn, "hopf" => p_hopf, "sl" => p_sl)

model = models[args["form"]]
stoch = stochs[args["form"]]
u0 = u0s[args["form"]]
p = psets[args["pset"]]
tspan = (0., args["stochtime"])
noises = eval(Meta.parse(args["noises"]))
signals = eval(Meta.parse(args["signals"]))

all_results = map(noises) do noise
    results = map(signals) do signal
        signal => sim_pop_ic(model, stoch, u0, [signal; p[2:end-3]; noise*0.25; noise*1.0; p[end]]; popsize=args["cells"], selectR=args["initstate"])
    end
    noise => Dict(results)
end
save(args["output"], "results", Dict(all_results))
