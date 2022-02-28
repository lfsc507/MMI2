using DifferentialEquations
using Statistics
using StatsBase
using Plots
using LinearAlgebra
using Distributions
using HDF5
using JLD2
using GaussianMixtures

using Logging
logger = SimpleLogger(stdout, Logging.Error) # Quiet GaussianMixtures output
global_logger(logger);

# Compute total mRNA at each timepoint in each cell (for MMI2-SSB state)
function reduce_mmi(sol)
    popsize = size(sol)[3]
    mT = sum(sol[2:5,:,:], dims=1)
    return reshape(mT[:,:,:], (size(sol)[2], popsize))
end

# Get mRNA 1 concentration at each timepoint in each cell (for repressilator or genetic oscillator state)
function reduce_repressilator(sim)
    return sim[1, :, :]
end
function reduce_geneosc(sim)
    return sim[1, :, :]
end

# Heatmap of state distributions at (final) timepoint across a range of signals
function perturbed_results_heatmap(results; reduction=reduce_mmi, time=nothing, signame="σR", statelabel="[Rtot]", maxrna=5, binwidth=0.25, kwargs...)
    sR_values = keys(results) |> collect |> sort
    bins = 0:binwidth:maxrna
    final_histograms = map(sR_values) do sR
        mTs = reduction(results[sR])
        if isnothing(time)
            states = mTs[end, :]
        else
            states = mTs[time, :]
        end
        fit(Histogram, states, bins).weights
    end
    matrix = hcat(final_histograms...)
    heatmap(sR_values, collect(bins)[begin:(end - 1)] .+ Float64(bins.step / 2), matrix; 
        xlabel=signame, ylabel=statelabel, colorbar_title="Cells", xminorticks=10, tick_direction=:out, kwargs...)
end

# Separation (D) of two modes in a one-dimensional dataset
function bimode_distinguishability(data)
    return bimode_distinguishability(GMM(2, reshape(data, length(data), 1)))
end
function bimode_distinguishability(model::GMM)
    # "Separation of means relative to their widths"
    # "D > 2 is required for a clean separation between the modes"
    # --Oleg Y. Gnedin, http://www-personal.umich.edu/~ognedin/gmm/gmm_user_guide.pdf
    μ = means(model)
    return abs(μ[1] - μ[2]) / sqrt(sum(covars(model)) / 2)
end

# Bayesian Information Criterion for a GMM fit to a 1D dataset
function bic(model::GMM, data)
    # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/mixture/_gaussian_mixture.py#L821-L823
    average_log_likelihood = avll(model, reshape(data, length(data), 1))
    nparams = length(means(model)) * 3 - 1 # mean, variance, weight (but weights must sum to 1)
    return nparams * log(length(data)) - 2 * average_log_likelihood * length(data)
end

# ΔBIC, the advantage of a 2-Gaussian fit over a 1-Gaussian fit for a 1D dataset
function bic_bimodal_advantage(data)
    data_reshaped = reshape(data, length(data), 1)
    bic1 = bic(GMM(1, data_reshaped), data_reshaped)
    bic2 = bic(GMM(2, data_reshaped), data_reshaped)
    return bic1 - bic2 # Positive if bimodality allows a better model
end

# Time index at which a population timecourse reached equilibrium
function equilibriumindex(results, savedat=0.1, reduction=reduce_mmi; stabletime=100.0, checktime=5.0, maxrna=10, binwidth=0.25, mainspecies=2)
    timesteps = collect(axes(results)[2])
    bins = 0:binwidth:maxrna
    histograms = map(t -> fit(Histogram, results[mainspecies, t, :], bins).weights, timesteps)
    stableindex = stabletime / savedat |> round |> Int
    final = dropdims(mean(hcat(histograms[stableindex:end]...); dims=2); dims=2)
    rmsds = map(t -> sqrt(mean((histograms[t] .- final) .^ 2.0)), timesteps)
    maxrmsd = maximum(rmsds[stableindex:end])
    mTs = reduction(results)
    totalmeans = dropdims(mean(mTs, dims=2); dims=2)
    equilibriummean = mean(totalmeans[stableindex:end])
    equilibriummaxmean = maximum(totalmeans[stableindex:end])
    equilibriumminmean = minimum(totalmeans[stableindex:end])
    ok_rmsd = false
    ok_mean = false
    candidate = missing
    for index in 2:stableindex
        cleared = false
        if rmsds[index] > maxrmsd * 1.1
            ok_rmsd = false
            cleared = true
        end
        if !(equilibriumminmean < totalmeans[index] < equilibriummaxmean)
            ok_mean = false
            cleared = true
        end
        if cleared
            candidate = missing
            continue
        end
        if rmsds[index] < maxrmsd / 2
            ok_rmsd = true
        end
        if sign(totalmeans[index - 1] - equilibriummean) != sign(totalmeans[index] - equilibriummean)
            ok_mean = true
        end
        if ok_rmsd && ok_mean
            if ismissing(candidate)
                candidate = index
            elseif (index - candidate) * savedat >= checktime
                return candidate
            end
        end
    end
    return candidate
end

# Timecourse of population state mean and state distribution for a population
function timecourse_heatmap(results; startindex=first(axes(results)[2]), endindex=last(axes(results)[2]), savedat=0.1, stabletime=100.0, reduction=reduce_mmi, statename="[Rtot]", maxrna=10, binwidth=0.25, kwargs...)
    mTs = reduction(results)
    timesteps = collect(startindex:endindex)
    bins = 0:binwidth:maxrna
    step_histograms = map(timesteps) do time
        fit(Histogram, mTs[time, :], bins).weights
    end
    matrix = hcat(step_histograms...)
    diagram = heatmap(timesteps * savedat, collect(bins)[begin:(end - 1)] .+ Float64(bins.step / 2), matrix; xlabel="Time", ylabel=statename, colorbar_title="Cells", kwargs...)
    means = dropdims(mean(mTs, dims=2); dims=2)
    plot!(diagram, (timesteps .- 1) * savedat, means[timesteps], color=:white, legend=nothing)
    stableindex = stabletime / savedat |> round |> Int
    hline!(diagram, [mean(mTs[stableindex:end, :]), maximum(means[stableindex:end]), minimum(means[stableindex:end])], color=:gray, legend=nothing)
    equilibriumstart = equilibriumindex(results, savedat, reduction; stabletime)
    vline!(diagram, [(equilibriumstart - 1) * savedat], color=:green, legend=nothing)
    xaxis!(diagram, xminorticks=10, tick_direction=:out, xlims=(0, (endindex - 1) * savedat))
    yaxis!(diagram, ylims=(0, maxrna))
end

# Create a wrapper function that runs the original several times and produces the minimum (counters nondeterminism in GMM fit)
function takemin(wrap, runs)
    function wrapper(args...)
        results = map(_ -> wrap(args...), 1:runs)
        return minimum(results)
    end
    return wrapper
end

# ΔBIC and separation line plot(s) for a set of signals (and different noise levels and initial conditions)
function perturbed_reestablishment_subplot(record, title; reduction=reduce_mmi, separation_lim=Inf, linekws=Dict())
    sR_values = keys(record) |> collect |> sort
    endpoints = map(sR_values) do sR
        mTs = reduction(record[sR])
        mTs[end, :]
    end
    distinguishabilities = map(takemin(bimode_distinguishability, 4), endpoints)
    bicadvantages = map(takemin(bic_bimodal_advantage, 4), endpoints)
    titlekws = Dict()
    if ~isnothing(title)
        titlekws = Dict(:title => title, :titlefontsize => 11)
    end
    subplot = plot(sR_values, bicadvantages; legend=nothing, ylabel="ΔBIC", yguidefontcolor=:blue, titlekws..., linekws...)
    hline!(subplot, [0.0], color=:darkblue, legend=nothing)
    twin = twinx(subplot)
    plot!(twin, sR_values, distinguishabilities; color=:orange, xticks=:none, legend=nothing, ylabel="Separation", yguidefontcolor=:orange, linekws...)
    hline!(twin, [2.0], color=:brown, legend=nothing)
    yaxis!(twin, ylims=(0, separation_lim))
    return subplot
end
function perturbed_reestablishment_plot(records; noiselabel="Noise", kwargs...)
    subplots = map(noise -> perturbed_reestablishment_subplot(records[noise], "$noiselabel = $noise"; kwargs...), keys(records) |> collect |> sort)
    plot(subplots..., layout=(length(keys(records)), 1), size=(600, 400), right_margin=15*Plots.mm)
end
function perturbed_reestablishment_plot(records::AbstractVector; noiselabel="Noise", size=(1000, 400), kwargs...)
    columns = map(run -> map(noise -> 
        perturbed_reestablishment_subplot(run[noise], isnothing(noiselabel) ? nothing : "$noiselabel = $noise"; kwargs...), 
        keys(run) |> collect |> sort), records)
    subplots = hcat(columns...)
    plot(permutedims(subplots)..., layout=grid(length(columns[1]), length(columns)), size=size, left_margin=5*Plots.mm, right_margin=15*Plots.mm)
end

# Plot time to equilibrium for a set of signals (and different noise levels and initial conditions)
function perturbed_equilibrium_plot(records::Dict{T, Array{T, 3}}; savedat=0.1, kwargs...) where T <: Real
    sR_values = keys(records) |> collect |> sort
    equilibriumtimes = map(sR -> (equilibriumindex(records[sR]) - 1) * savedat, sR_values)
    plot(sR_values, equilibriumtimes, xlabel="sR", ylabel="Time", legend=nothing; kwargs...)
end
function perturbed_equilibrium_plot(records::Dict{T, Dict{T, Array{T, 3}}}; kwargs...) where T <: Real
    noises = keys(records) |> collect |> sort
    subplots = map(noise -> perturbed_equilibrium_plot(records[noise]; title="Noise = $noise", titlefontsize=11, kwargs...), noises)
    plot(subplots..., layout=(length(noises), 1))
end
function perturbed_equilibrium_plot(records::AbstractVector; kwargs...)
    columns = map(run -> map(noise -> perturbed_equilibrium_plot(run[noise]; title="Noise = $noise", titlefontsize=11, kwargs...), keys(run) |> collect |> sort), records)
    subplots = hcat(columns...)
    plot(permutedims(subplots)..., layout=grid(length(columns[1]), length(columns)), link=:all, size=(1000, 450), left_margin=5*Plots.mm, bottom_margin=5*Plots.mm)
end

# Print maximum and median equilibrium times for populations in a range of signals (missing or 1000 if ET is too long to determine)
function eqtimestats(desc, records, span)
    eqtimes = map(sR -> (equilibriumindex(records[sR]) - 1) * 0.1, span)
    filled = replace(eqtimes, missing => 1000.0)
    println(desc, "\tMax: ", maximum(eqtimes), "\t", "Avg: ", median(filled))
end

# Load a set of loose Python Gillespie results (filenames under "gillespie" directory, specified by regex) into a noise -> signal -> array dict to match Julia SDE outputs
function loadgillespies(re)
    results = Dict{Float64, Dict{Float64, Array{Float64, 3}}}()
    for filename in filter(f -> match(re, f) !== nothing, readdir("gillespie"))
        file = h5open("gillespie/$filename")
        noise = read(attributes(file)["V"])
        if noise ∉ keys(results)
            results[noise] = Dict{Float64, Array{Float64, 3}}()
        end
        signal = read(attributes(file)["s_m"])
        data = read(file["results"])
        results[noise][signal] = permutedims(data, reverse(1:ndims(data)))
        close(file)
    end
    return results
end
