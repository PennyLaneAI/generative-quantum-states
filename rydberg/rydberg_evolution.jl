# Original code is generously provided by Dr. Mao Lin (Amazon Braket) 
using Distributed
using ProgressBars
using DelimitedFiles
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dim"
        help = "lattice dimension"
        arg_type = Int
        default = 2
        "--nx"
        help = "lattice width for 2D lattice; #qubits for 1D lattice"
        arg_type = Int
        default = 5
        "--ny"
        help = "lattice height"
        arg_type = Int
        default = 3
        "--total_time"
        help = "evolution time"
        arg_type = Float64
        default = 18.0
        "--time_steps"
        help = "# time steps"
        arg_type = Int
        default = 16
        "--use_cuda"
        help = "use CUDA"
        action = :store_true
        default = false
        "--n_threads"
        arg_type = Int
        default = -1
        "--itr_init"
        help = "the initial value of the interaction_range"
        arg_type = Float64
        default = 1.05
        "--itr_step"
        help = "the step size of the interaction_range"
        arg_type = Float64
        default = 0.1
        "--itr_final"
        help = "the initial value of the interaction_range"
        arg_type = Float64
        default = 2.45
        "--n_shots"
        help = "the number of measurement snapshots"
        arg_type = Int
        default = 1000
        "--data_folder"
        arg_type = String
        default = "../data/rydberg/"
        "--solver"
        arg_type = String
        default = "Vern8"
        "--detuning_start_measure"
        arg_type = Float64
        default = -2.0
        help = "the initial detuning value to measure"
        "--blockade_subspace"
        arg_type = Bool
        default = true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
cpu_count = length(Sys.cpu_info())
interaction_ranges = parsed_args["itr_init"]:parsed_args["itr_step"]:parsed_args["itr_final"]
println("interaction_ranges: ", interaction_ranges)
if parsed_args["n_threads"] < 0
    max_cpu_threads = length(interaction_ranges)
else
    max_cpu_threads = parsed_args["n_threads"]
end
if nprocs() == 1
    n_cpu_threads = min(cpu_count - 1, max_cpu_threads)
    n_cpu_threads = min(length(interaction_ranges), n_cpu_threads)
    println("#threads: ", n_cpu_threads)
    if n_cpu_threads > 0
        addprocs(n_cpu_threads; exeflags=`--project=$(Base.active_project())`)
    end
end
@everywhere begin
    using Bloqade
    using StatsBase
    using BitBasis

    using NPZ
    using ProgressBars
    worker_id = Distributed.myid()
    args = $parsed_args
    dim = args["dim"]
    @assert dim <= 2
    nx = args["nx"]
    ny = args["ny"]
    if dim == 2
        ny = args["ny"]
    else
        ny = 1
    end
    n_qubits = nx * ny

    use_CUDA = args["use_cuda"]
    if use_CUDA
        using CUDA, Adapt
    end
    solver = getfield(Main, Symbol(args["solver"]))
    total_time = args["total_time"]
    time_steps = args["time_steps"]
    base_total_t = 3.5
    base_t1 = 0.2
    time_ratio = base_total_t / total_time

    if dim == 2
        Ωmax = 2π * 4.3
        Δmin = -2π * 15
        Δmax = 2π * 20
    elseif dim == 1
        Ωmax = 2π * 5
        Δmin = -2π * 10
        Δmax = 2π * 15
    else
        throw(DomainError())
    end


    t1 = base_t1 / time_ratio
    t2 = base_total_t / time_ratio
    t_range = t2 - t1

    detuning_1 = Δmin / Ωmax
    detuning_2 = Δmax / Ωmax
    detuning_range = detuning_2 - detuning_1
    if (args["detuning_start_measure"] > detuning_1) && (args["detuning_start_measure"] < detuning_2)
        detuning_start_measure = args["detuning_start_measure"]
    else
        detuning_start_measure = detuning_1
    end

    t_start_measure = t1 + t_range * (detuning_start_measure - detuning_1) / detuning_range
    @assert t_start_measure >= t1
    @assert t_start_measure <= t2
    # ratio_start_measure = -2.0 # If Δ/Ω < this ratio, we do not perform any measurement
    sweep_rate = (Δmax - Δmin) / ((base_total_t - base_t1) / time_ratio)
    Δ = piecewise_linear(clocks=[0.0, t1, t2], values=[Δmin, Δmin, Δmax])
    Ω = piecewise_linear(clocks=[0.0, t1, t2], values=[0.0, Ωmax, Ωmax])

    C6 = 2π * 862690
    Rb = (C6 / Ωmax)^(1 / 6)
    times = Vector{Float64}()
    for t in range(t_start_measure, stop=t2, length=time_steps)
        # t = round(t, digits=6)
        Δ_t = Δ.f(t)
        Ω_t = Ω.f(t)
        push!(times, t)
    end

    function generate_lattice(; lattice_const::Float64, dim::Int, nx::Int, ny::Int)
        if dim == 1
            atoms = generate_sites(ChainLattice(), nx, scale=lattice_const)
        elseif dim == 2
            atoms = generate_sites(SquareLattice(), nx, ny, scale=lattice_const)
        end
        return atoms
    end

    function evolve_and_measure(atoms; n_shots::Int=1000, interaction_range::Float64=nothing)
        """For a given lattice const, return the evolution of the state"""
        t0 = time()

        # println("--------Start Blockade Subspace--------")
        if args["blockade_subspace"]
            subspace = blockade_subspace(atoms, Rb)
            reg = zero_state(subspace)
        else
            reg = zero_state(n_qubits)
        end
        # println("Time: $(round(time() - t0,digits=3)) seconds")
        t0 = time()
        # println("--------Construct Hamilnian--------")
        h = rydberg_h(atoms; Δ=Δ, Ω=Ω)
        duration = Δ.duration
        # println("Time: $(round(time() - t0,digits=3)) seconds")
        t0 = time()
        # println("--------Construct Prob--------")
        prob = SchrodingerProblem(reg, duration, h, progress=true)
        if use_CUDA
            prob = adapt(CuArray, prob)
        end
        # println("Time: $(round(time() - t0,digits=3)) seconds")
        t0 = time()
        # println("--------Init Integrator--------")
        integrator = init(prob, solver())
        # println("Time: $(round(time() - t0,digits=3)) seconds")
        z_densities, x_densities = [], []
        x_corrs, y_corrs, z_corrs, n_corrs = [], [], [], []
        measurements = []
        Δ_list, Ω_list, t_list = Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
        if worker_id == 2
            iterator = tqdm(times, leave=false)
            set_description(iterator, "iter")
        else
            iterator = times
        end
        # println("--------Start Simulation--------")
        for iter in TimeChoiceIterator(integrator, iterator)
            # t0 = time()
            t = iter[2] # time
            push!(t_list, t)
            push!(Δ_list, Δ.f(t))
            push!(Ω_list, Ω.f(t))
            if use_CUDA
                freg = adapt(Array, prob.reg)
            else
                freg = reg
            end
            normalize!(freg)
            push!(z_densities, rydberg_density(freg))
            push!(measurements, measure(freg, nshots=n_shots))

            # time_duration = round((time() - t0) / 60.0, digits=2)
            # println("Measurement duration: $time_duration min")
        end
        result = Dict(
            "density_z" => hcat(z_densities...)',
            "measurements" => map(bint, hcat(measurements...)'),
            "Omega" => Ω_list,
            "Delta" => Δ_list,
            "detuning" => Δ_list ./ Ω_list,
            "interaction_range" => interaction_range,
            "time" => t_list,
            "total_time" => total_time,
            "sweep_rate" => sweep_rate,
            "rel_time" => t_list ./ total_time,
        )
        return result, z_densities
    end

    function get_order_parameter(densities, which)
        """For a given evolution of rydberg densities, and an order (which="Z2" or "Z3"), return the order parameter"""

        order = Vector{Float64}()
        for density in densities
            if which == "checkboard"
                push!(order, sum(density[1:2:end-1]) / sum(density[1:end-1]))
            else
                throw(MethodError("$which" * " is not a supported order parameter"))
            end
        end

        return order
    end

    function get_order_parameter_2(densities)
        """For a given evolution of rydberg densities, return if they are Z2 or Z3 order, or disorder"""
        order2 = get_order_parameter(densities, "checkboard")
        order_params = Dict("checkboard" => order2,)
        return order_params
    end

    function job(interaction_range)
        """Distribute the tasks to different cores"""
        lattice_const = Rb / interaction_range
        atoms = generate_lattice(; dim=dim, nx=nx, ny=ny, lattice_const=lattice_const)
        result, z_densities = evolve_and_measure(atoms; n_shots=args["n_shots"], interaction_range=interaction_range)
        order_params = get_order_parameter_2(z_densities)
        merge!(result, order_params) # Write order_params to result
        rounded_intr_range = round(interaction_range, digits=2)
        npzwrite(folder * "/interaction_range-$rounded_intr_range" * ".npz", result) # save result
        return
    end

    folder = "$(args["data_folder"])/$(dim)D-Phase_$(nx)"
    if dim == 2
        folder = folder * "x$(ny)"
    end
    folder = folder * "/$(total_time)µs/"
end

println("nx = $nx")
println("ny = $ny")
mkpath(folder) # make directory

t0 = time()
exp_iterator = ProgressBar(interaction_ranges)
set_description(exp_iterator, "Exp")
if nprocs() == 1
    for i in exp_iterator
        job(i)
    end
else
    pmap(job, exp_iterator)
end

time_duration = round((time() - t0) / 60.0, digits=2)
println("Time $time_duration min")

npzwrite(folder * "interaction_ranges.npz", interaction_ranges)