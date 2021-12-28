using DifferentialEquations
using LSODA
using JLD2
using ArgParse
using StatsBase
using Distributions
using ThreadTools

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
    "--noises"
        help = "noise values"
        default = "0.5:0.5:1.5"
    "--stochtime"
        help = "stochastic simulation endpoint"
        arg_type = Float64
        default = 50.0
    "--initcells"
        help = "number of cells selected"
        arg_type = Int
        default = 4
end
args = parse_args(parser)

cellspecs = Dict("add" => 9, "mul" => 12)
const cellspec = cellspecs[args["form"]]
const divlimit = 75.0

function model_mmi2_add(du, u, p, t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, kG, σr, σR, σG, τ, V = p
    r, R, c1A, c1B, c2 = u[1:cellspec:end], u[2:cellspec:end], u[3:cellspec:end], u[4:cellspec:end], u[5:cellspec:end]
    sig = sR

    @. du[1:cellspec:end] = sr - kr * r + koff * (c1A+c1B) - K1 * koff * 2R * r +
                koff * 2c2 - K2 * koff * (c1A+c1B) * r +
                (c1A+c1B) * kR * a1 + 2c2 * kR * a2 + u[7:cellspec:end]

    @. du[2:cellspec:end] = sig + sR0 - kR * R + koff * (c1A+c1B) - K1 * koff * 2R * r +
                (c1A+c1B) * kr * b1 + u[8:cellspec:end]

    @. du[3:cellspec:end] = K1 * koff * R * r - koff * c1A +
                koff * c2 - K2 * koff * c1A * r +
                c2 * 1kr * b2 - c1A * kR * a1 - c1A * kr * b1

    @. du[4:cellspec:end] = K1 * koff * R * r - koff * c1B +
                koff * c2 - K2 * koff * c1B * r +
                c2 * 1kr * b2 - c1B * kR * a1 - c1B * kr * b1

    @. du[5:cellspec:end] = K2 * koff * (c1A+c1B) * r - koff * 2c2 -
                c2 * 2kr * b2 - c2 * kR * a2

    @. du[6:cellspec:end] = kG + u[9:cellspec:end]
    
    for i in 7:9
        du[i:cellspec:end] = -u[i:cellspec:end]/τ
    end
end

function stoch_mmi2_add(du, u, p, t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, kG, σr, σR, σG, τ, V = p
    for i in 1:6
        du[i:cellspec:end] .= 0
    end
    du[7:cellspec:end] .= sqrt(2.0*σr^2.0/τ)
    du[8:cellspec:end] .= sqrt(2.0*σR^2.0/τ)
    du[9:cellspec:end] .= sqrt(2.0*σG^2.0/τ)
end

function model_mmi2_mul(du, u, p, t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, kG, σr, σR, σG, τ, V = p
    r, R, c1A, c1B, c2 = u[1:cellspec:end], u[2:cellspec:end], u[3:cellspec:end], u[4:cellspec:end], u[5:cellspec:end]
    sig = sR

    @. du[1:cellspec:end] = sr - kr * r + koff * (c1A+c1B) - K1 * koff * 2R * r +
                koff * 2c2 - K2 * koff * (c1A+c1B) * r +
                (c1A+c1B) * kR * a1 + 2c2 * kR * a2 + u[7:cellspec:end]

    @. du[2:cellspec:end] = sig + sR0 - kR * R + koff * (c1A+c1B) - K1 * koff * 2R * r +
                (c1A+c1B) * kr * b1 + u[8:cellspec:end]

    @. du[3:cellspec:end] = K1 * koff * R * r - koff * c1A +
                koff * c2 - K2 * koff * c1A * r +
                c2 * 1kr * b2 - c1A * kR * a1 - c1A * kr * b1 + u[9:cellspec:end]

    @. du[4:cellspec:end] = K1 * koff * R * r - koff * c1B +
                koff * c2 - K2 * koff * c1B * r +
                c2 * 1kr * b2 - c1B * kR * a1 - c1B * kr * b1 + u[10:cellspec:end]

    @. du[5:cellspec:end] = K2 * koff * (c1A+c1B) * r - koff * 2c2 -
                c2 * 2kr * b2 - c2 * kR * a2 + u[11:cellspec:end]

    @. du[6:cellspec:end] = kG + u[12:cellspec:end]
    
    for i in 7:12
        du[i:cellspec:end] = -u[i:cellspec:end]/τ
    end
end

function stoch_mmi2_mul(du, u, p, t)
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, kG, σr, σR, σG, τ, V = p
    for i in 1:6
        du[i:cellspec:end] .= 0
    end
    for i in 7:11
        du[i:cellspec:end] = sqrt(2.0*σR^2.0/τ) * u[(i - 6):cellspec:end]
    end
    du[12:cellspec:end] .= sqrt(2.0*σG^2.0/τ)
end

function condition_mmi2_split(u, t, integrator)
    for i in 1:5
        minconc = minimum(u[i:cellspec:end])
        if minconc < -0.01 # Adjust if needed
            println("Possibly diverging conc of $minconc for species $i at t = $t")
            terminate!(integrator)
        end
    end
    return maximum(u[6:cellspec:end]) >= divlimit
end

function affect_mmi2_split(integrator)
    #println("Affecting at t = $(integrator.t)")
    length(integrator.u) > 5000 * cellspec && error("Runaway division")
    sR, a1, a2, b1, b2, kR, kr, K1, K2, koff, sr, sR0, kG, σr, σR, σG, τ, V = integrator.p
    while maximum(integrator.u[6:cellspec:end]) >= divlimit
        prevend = length(integrator.u)
        maxcell = argmax(integrator.u[6:cellspec:end])
        maxbase = (maxcell - 1) * cellspec
        resize!(integrator, prevend + cellspec)
        u = integrator.u
        u[maxbase + 6] >= divlimit || error("Index $(maxbase + 6) should be above division threshold")
        u[maxbase + 6] = 0
        for spec in 1:5
            conc = u[maxbase + spec]
            #print("Species $spec conc was $conc, ")
            if conc > 0
                dist = truncated(Normal(0.5, 0.5 / sqrt(2 * conc * V)), 0., 1.)
                newconc = 2 * rand(dist) * conc
                u[end - cellspec + spec] = newconc
                u[maxbase + spec] = 2 * conc - newconc
                #println("now $newconc and $(2 * conc - newconc)")
            else
                #println("unchanged (< 0)")
                u[end - cellspec + spec] = conc
            end
        end
        u[(end - cellspec + 7):end] = u[(maxbase + 7):(maxbase + cellspec)]
        #println("Split cell $maxcell, now $(length(u) / cellspec |> Int) cells")
    end
    return nothing
end

callback_mmi2_split = DiscreteCallback(condition_mmi2_split, affect_mmi2_split)

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
    divlimit / 6.0, #13 kG
    0.25, #14 σr    replaced
    1.00, #15 σR    replaced
    1.00, #16 σG    replaced
    0.1, #17 τ
    2000.0 #18 V
   ];

p_hopf = [ # Hopf only, like Figure 3B
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
    divlimit / 6.0, #13 kG
    0.25, #14 σr    replaced
    1.00, #15 σR    replaced
    1.00, #16 σG    replaced
    0.1, #17 τ
    2000.0 #18 V
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
    divlimit / 6.0, #13 kG
    0.25, #14 σr    replaced
    1.00, #15 σR    replaced
    1.00, #16 σG    replaced
    0.1, #17 τ
    2000.0 #18 V
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
    divlimit / 6.0, #13 kG
    0.25, #14 σr    replaced
    1.00, #15 σR    replaced
    1.00, #16 σG    replaced
    0.1, #17 τ
    2000.0 #18 V
  ];

function timecourse_array(sols, savevars, savedat=0.1)
    timepoints = 0:savedat:sols[end].t[end]
    maxcells = map(s -> length(s[end]) / cellspec, sols) |> maximum |> Int
    timecourse = fill!(Array{Float64, 3}(undef, length(savevars), length(timepoints), maxcells), NaN)
    for (tindex, time) in enumerate(timepoints)
        for (sindex, sol) in enumerate(sols)
            (time < sol.t[begin] || time > sol.t[end]) && continue
            state = sol(time)
            curcells = length(state) / cellspec |> Int
            timecourse[:, tindex, :] .= NaN
            for (target, source) in enumerate(savevars)
                timecourse[target, tindex, 1:curcells] = state[source:cellspec:end]
            end
        end
    end
    return timecourse
end

function sim_pop(model, model_stoch, u0, pars, initcells; selectR::String="lo", saveat=0.1)
    selectR == "lo" || selectR == "hi" || error("invalid selectR")
    pars_nogrowth = copy(pars)
    pars_nogrowth[13] = 0 # kG = 0
    prob_ode = ODEProblem(model, u0, (0., 200.), pars_nogrowth);
    r0s = range(0.001, 10.0, length=20)
    R0s = range(0.001, 10.0, length=20)
    function prob_func_cells_init(prob,i,repeat)
        m, n = i ÷ length(r0s), i % length(r0s)
        if n != 0
            r0, R0 = r0s[m+1], R0s[n]
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
    u0_single = sol_init[:, idsel[1], idsel[2]]
    println("Starting stochastic simulation with initial condition ", u0_single)
    u0_cells = repeat(u0_single, initcells)
    u0_cells[6:cellspec:end] = rand(Uniform(0, divlimit), initcells)
    prob = SDEProblem(model, model_stoch, u0_cells, tspan, pars)
    sol = solve(prob, callback=callback_mmi2_split, saveat=saveat, dt=0.001, maxiter=1e12);
    sols = [sol]
    while sols[end].t[end] < tspan[2]
        println("Interrupted at t = $(sols[end].t[end]) with status $(sol.retcode)")
        if sol.retcode == :Terminated
            fullcourse = timecourse_array(sols, 1:cellspec, saveat)
            stopindex = size(fullcourse)[2] # Adjust savepoint selection if needed
            while minimum(filter(!isnan, fullcourse[1:5, stopindex, :])) < -0.0005
                stopindex -= 1
            end
            stopindex -= 1
            trest = ((stopindex - 1) * saveat, tspan[2])
            println("Using savepoint t = $(trest[1])")
            ulast = filter(!isnan, fullcourse[:, stopindex, :])
        else
            stopindex = length(sols[end])
            while !isinteger(round(sols[end].t[stopindex] / saveat; digits=5))
                stopindex = stopindex - 1
            end
            println("Using endpoint t = $(sols[end].t[stopindex])")
            ulast = sols[end].u[stopindex]
            trest = (sols[end].t[stopindex], tspan[2])
        end
        prob_resume = remake(prob, u0=ulast, tspan=trest)
        sol = solve(prob_resume, callback=callback_mmi2_split, saveat=saveat, dt=0.001, maxiter=1e12)
        push!(sols, sol)
    end
    return timecourse_array(sols, 1:5, saveat)
end

u0s = Dict("add" => [0.8, 0.1, 0., 0., 0., 0., 0., 0., 0.], "mul" => [0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
models = Dict("add" => model_mmi2_add, "mul" => model_mmi2_mul)
stochs = Dict("add" => stoch_mmi2_add, "mul" => stoch_mmi2_mul)
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
        signal => sim_pop(model, stoch, u0, [signal; p[2:end-5]; noise*0.25; noise; noise; p[(end-1):end]], args["initcells"]; selectR=args["initstate"])
    end
    noise => Dict(results)
end
save(args["output"], "results", Dict(all_results))
