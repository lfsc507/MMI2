using ArgParse
using JLD2

parser = ArgParseSettings(allow_ambiguous_opts=false)
@add_arg_table! parser begin
    "main"
        required = true
    "patch"
        required = true
    "--overwrite"
        arg_type = Bool
        default = true
end
args = parse_args(parser)

results = load(args["main"])["results"];
patch = load(args["patch"])["results"];
for (noise, sigscan) in patch
    for (signal, data) in sigscan
        (signal in keys(results[noise])) == args["overwrite"] || error("check overwrite for results[$noise][$signal]")
        results[noise][signal] = data
        println("including results[$noise][$signal]")
    end
end
save(args["main"], "results", results)
