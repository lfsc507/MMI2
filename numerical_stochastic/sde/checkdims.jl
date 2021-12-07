using JLD2

expected = parse(Int, ARGS[1])

for input in ARGS[2:end]
    println(input)
    data = load(input)["results"];
    for noise in keys(data)
        for signal in keys(data[noise])
            shape = size(data[noise][signal])
            if shape[2] < expected
                println("$noise $signal ", shape)
            end
        end
    end
    println()
end
