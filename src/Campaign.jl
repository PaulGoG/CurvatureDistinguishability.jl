"""
Supervised execution of pipeline configurations: the wiring between the
generic process supervisor (`Supervision`) and a pipeline worker
(`scripts/run_pipeline.jl`) that publishes a heartbeat, checkpoints its work
items and continues an interrupted run directory.
"""
module Campaign

using DocStringExtensions: TYPEDSIGNATURES
using ..Config: PipelineSettings, load_and_validate_config
using ..Provenance: resolve_run_dir, unique_run_dir, run_id_from_config,
    completed_stages, abandoned_stages, read_run_metadata, stage_key, backup_existing!
using ..Supervision: Supervision, SupervisionResult, WorkerSpec, supervise
using ..Checkpoint: count_items

export supervise_pipeline, requested_stages

"""
$(TYPEDSIGNATURES)

Stage keys (`"sweep:<name>"`, `"map:<name>"`) the configuration asks for, in
execution order.
"""
function requested_stages(cfg::PipelineSettings)
    stages = String[]
    cfg.run_1d_sweeps && append!(stages, (stage_key(:sweep, s.name) for s in cfg.sweeps))
    cfg.run_2d_mapping && append!(stages, (stage_key(:map, m.name) for m in cfg.maps))
    return stages
end

# files of a run directory whose growth shows that the worker is advancing
function progress_files(run_dir::AbstractString)
    files = [joinpath(run_dir, "run.log")]
    for (kind, names) in (("sweeps", ("sweep.log", "checkpoint.csv")),
        ("maps", ("mapping.log",)))
        root = joinpath(run_dir, kind)
        isdir(root) || continue
        for stage in readdir(root), name in names
            push!(files, joinpath(root, stage, name))
        end
    end
    return files
end

# finished work items of a run directory: checkpointed separations plus
# completed stages (a map counts once, when it completes)
function finished_items(run_dir::AbstractString)
    items = length(completed_stages(run_dir))
    root = joinpath(run_dir, "sweeps")
    isdir(root) || return items
    return items + sum(
        count_items(joinpath(root, stage, "checkpoint.csv"))
        for stage in readdir(root); init = 0)
end

function run_status(run_dir::AbstractString, stages::Vector{String})
    done = completed_stages(run_dir)
    all(in(done), stages) && return :complete
    haskey(read_run_metadata(run_dir), "finished") || return :incomplete
    settled = union(done, abandoned_stages(run_dir))
    return all(in(settled), stages) ? :partial : :incomplete
end

heap_flags(gb::Real) =
    gb <= 0 ? String[] :
    gb >= 1 ? ["--heap-size-hint=$(round(Int, gb))G"] :
    ["--heap-size-hint=$(max(1, round(Int, 1024gb)))M"]

"""
$(TYPEDSIGNATURES)

Run the configuration at `config_path` to completion under the watchdog and
retry budget of its `[supervision]` table. The worker is
`julia worker_script --config … --run-dir … --heartbeat-period …`, relaunched
into the same run directory after a hang, a crash or a failed stage; finished
stages and work items are never recomputed. Returns `(run_dir, result)` with a
`Supervision.SupervisionResult`; a configuration whose run is already complete
returns at once with status `:complete` and no attempts.

Keywords: `worker_script` (path of `scripts/run_pipeline.jl`), `julia` (binary,
default: the running one), `threads` (worker thread specification, default
`"auto"`), `fresh = false` (start a new sibling directory even when an
unfinished run exists), `worker_env` (environment overrides of the worker,
e.g. a device-visibility mask; `nothing` removes a variable).

Configuration errors propagate as `ArgumentError` before anything is launched.
"""
function supervise_pipeline(config_path::AbstractString, project_root::AbstractString,
    output_dir::AbstractString; worker_script::AbstractString,
    julia::AbstractString = Base.julia_cmd().exec[1], threads::AbstractString = "auto",
    fresh::Bool = false, worker_env = Pair{String,Union{Nothing,String}}[])
    config_path = abspath(config_path)
    cfg = load_and_validate_config(config_path)
    base_dir = joinpath(project_root, output_dir)
    run_dir, state =
        fresh ? (unique_run_dir(base_dir, run_id_from_config(config_path)), :fresh) :
        resolve_run_dir(base_dir, config_path)
    state === :complete &&
        return run_dir, SupervisionResult(:complete, Supervision.AttemptRecord[], String[])

    stages = requested_stages(cfg)
    ledger = joinpath(run_dir, "supervision.toml")
    # a new supervisor session starts a new ledger; stages abandoned by an
    # earlier session are tried again
    isfile(ledger) && backup_existing!(ledger)
    settings = cfg.supervision
    function command(attempt::Int)
        cmd = `$julia --startup-file=no --threads=$threads $(heap_flags(cfg.heap_size_hint_gb)) $worker_script --config $config_path --output-dir $output_dir --run-dir $run_dir --heartbeat-period $(settings.poll_interval_s / 2)`
        return isempty(worker_env) ? cmd : addenv(cmd, worker_env...)
    end
    function abandon_stalled!()
        settled = union(completed_stages(run_dir), abandoned_stages(run_dir))
        k = findfirst(!in(settled), stages)
        return k === nothing ? nothing : stages[k]
    end
    spec = WorkerSpec(; command,
        console_log = attempt ->
            joinpath(run_dir, "logs", "attempt_$(attempt).console.log"),
        heartbeat_path = joinpath(run_dir, "heartbeat.toml"),
        progress_paths = () -> progress_files(run_dir),
        items_done = () -> finished_items(run_dir),
        status = () -> run_status(run_dir, stages), abandon_stalled!)
    return run_dir, supervise(spec, settings; ledger_path = ledger)
end

end # module
