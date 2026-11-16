"""
Run identity and safekeeping: canonical configuration hashing, config
snapshots, backup-before-overwrite semantics and run metadata.
"""
module Provenance

using DocStringExtensions: TYPEDSIGNATURES
using Dates: now
using FileWatching.Pidfile: mkpidlock, PidlockedError
using SHA: sha256
using TOML: TOML
using DrWatson: gitdescribe
using InteractiveUtils: versioninfo
using LinearAlgebra: BLAS

export run_id_from_config, effective_config, identity_config,
    unique_run_dir, resolve_run_dir, snapshot_config, snapshot_manifest,
    backup_existing!, write_run_metadata, read_run_metadata,
    write_hardware_fingerprint, git_state, package_version, check_package_version,
    stage_key, completed_stages, mark_stage_complete!, abandoned_stages,
    note_figure_failure!
public acquire_run_lock, record_attempt!

# top-level key naming the base file an overlay configuration is merged onto
const BASE_CONFIG_KEY = "base_config"

"""
Configuration sections that describe how a run executes rather than what it
computes: backend and concurrency (`[hardware]`), memory budgets
(`[safety]`), diagnostics (`[monitoring]`) and the watchdog of supervised runs
(`[supervision]`). They are excluded from the run identifier and recorded in
`metadata.toml` and `hardware.txt` instead.
"""
const EXECUTION_SECTIONS = ("hardware", "safety", "monitoring", "supervision")

# ledger written by the supervisor into the run directory
const SUPERVISION_LEDGER = "supervision.toml"

# lock files of a run directory: the pipeline worker and the supervisor
const RUN_LOCK = "run.pid"
const SUPERVISOR_LOCK = "supervisor.pid"

"""
$(TYPEDSIGNATURES)

Parsed configuration with its base file resolved. A top-level
`base_config = "file.toml"` (path relative to the overlay's directory)
names a base whose tables are deep-merged beneath the overlay's: sub-tables
recurse, while scalars, arrays and arrays of tables (`[[sweeps]]`,
`[[maps]]`) present in the overlay replace the base's. One level only: a
base that itself declares `base_config` is an error, as is a missing base.
The `base_config` key is stripped, so the result is the self-contained
table every consumer (validation, run-ID hashing, snapshots) operates on; a
file without `base_config` parses as-is.
"""
function effective_config(config_path::AbstractString)
    config = TOML.parsefile(config_path)
    haskey(config, BASE_CONFIG_KEY) || return config
    base_rel = pop!(config, BASE_CONFIG_KEY)
    base_rel isa AbstractString || throw(
        ArgumentError(
            "$BASE_CONFIG_KEY in $config_path must be a file path, got $(repr(base_rel))",
        ),
    )
    base_path = normpath(joinpath(dirname(abspath(config_path)), base_rel))
    isfile(base_path) ||
        throw(
            ArgumentError(
                "$BASE_CONFIG_KEY of $config_path names a missing file: $base_path",
            ),
        )
    base = TOML.parsefile(base_path)
    haskey(base, BASE_CONFIG_KEY) && throw(
        ArgumentError(
            "base configuration $base_path declares $BASE_CONFIG_KEY itself; " *
            "only one overlay level is supported",
        ),
    )
    return merge_config(base, config)
end

"""
$(TYPEDSIGNATURES)

The part of a parsed configuration that defines the computed problem: every
section except the [`EXECUTION_SECTIONS`](@ref). Two runs with equal identity
tables compute the same physical case with the same numerical method,
whatever hardware executes them.
"""
function identity_config(config::AbstractDict)
    identity = Dict{String,Any}(config)
    for section in EXECUTION_SECTIONS
        delete!(identity, section)
    end
    return identity
end

"""
Deep merge of `overlay` into `base`: sub-tables recurse; scalars, arrays and
arrays of tables in the overlay replace the base value. Neither input is
modified.
"""
function merge_config(base::AbstractDict, overlay::AbstractDict)
    merged = Dict{String,Any}(base)
    for (key, value) in overlay
        merged[key] =
            (value isa AbstractDict && get(merged, key, nothing) isa AbstractDict) ?
            merge_config(merged[key], value) : value
    end
    return merged
end

"""
$(TYPEDSIGNATURES)

Deterministic run identifier: the first 8 hex characters of the SHA-256 of
the canonically serialized identity table of the effective configuration
([`effective_config`](@ref) with the [`EXECUTION_SECTIONS`](@ref) removed by
[`identity_config`](@ref); keys sorted, values only, never the wall
clock). Identical physical/numerical content therefore maps to identical
IDs whatever hardware executes it; reruns on other machines land in
suffixed sibling directories whose `metadata.toml` records backend, host
and timings. Comments, formatting, execution settings and the split
between a base file and its overlay do not affect a run's identity.
"""
function run_id_from_config(config_path::AbstractString)
    identity = identity_config(effective_config(config_path))
    canonical = sprint(io -> TOML.print(io, identity; sorted = true))
    return "run_" * first(bytes2hex(sha256(canonical)), 8)
end

"""
$(TYPEDSIGNATURES)

Create and return an output directory for `run_id` under `base_dir`. If the
directory already exists (a rerun of the same configuration), a `_r2`,
`_r3`, … suffix is appended; existing results are never overwritten.
"""
function unique_run_dir(base_dir::AbstractString, run_id::AbstractString)
    mkpath(base_dir)
    dir = joinpath(base_dir, run_id)
    k = 1
    while true
        # mkdir fails on an existing directory, so two launches of one
        # configuration can never end up sharing a run directory
        try
            mkdir(dir)
            return dir
        catch err
            (err isa Base.IOError && ispath(dir)) || rethrow()
        end
        k += 1
        dir = joinpath(base_dir, "$(run_id)_r$k")
    end
end

"""
$(TYPEDSIGNATURES)

Version of this package as declared in its `Project.toml`; recorded in
`metadata.toml` as `package_version` and compared before a run directory is
continued ([`check_package_version`](@ref)).
"""
package_version() = string(pkgversion(@__MODULE__))

# whether the running package may continue the outputs in `run_dir`: its
# metadata records the same version, or there is no metadata yet (nothing
# has been computed)
function same_package_version(run_dir::AbstractString)
    isfile(joinpath(run_dir, "metadata.toml")) || return true
    return get(read_run_metadata(run_dir), "package_version", nothing) == package_version()
end

function version_mismatch_message(run_dir::AbstractString)
    recorded = get(read_run_metadata(run_dir), "package_version", nothing)
    origin =
        recorded === nothing ? "a package version that predates the record" :
        "package version $recorded"
    return "$run_dir was produced by $origin; version $(package_version()) " *
           "does not continue it"
end

"""
$(TYPEDSIGNATURES)

Throw an `ArgumentError` unless the outputs in `run_dir` may be continued by
the running package: its `metadata.toml` must record the same
`package_version`, or be absent (nothing has been computed yet). The run
identifier hashes the configuration only, so a release that changes the
scientific model maps an unchanged configuration onto the same identifier;
this check keeps such a release from completing or reusing a run begun
under another version, or under one that predates the record.
"""
function check_package_version(run_dir::AbstractString)
    same_package_version(run_dir) ||
        throw(ArgumentError(version_mismatch_message(run_dir) * "; start a new run"))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Run directory for `config_path` under `base_dir` when an interrupted run is to
be continued. The siblings `run_<id>`, `run_<id>_r2`, … are searched in order
for one whose `config.toml` snapshot equals the effective configuration,
execution sections included (so variants that share an identifier, such as
another Hessian chunk or another backend, never continue each other), and
whose outputs were produced by the running package version
([`check_package_version`](@ref)); a sibling that matches the configuration
only is skipped with a warning. Returns `(directory, state)` with `state`

  - `:complete`: that run finished with every stage complete,
  - `:resume`: that run is unfinished, or finished with failed or abandoned stages,
  - `:fresh`: no sibling matches; a new one was created ([`unique_run_dir`](@ref)).
"""
function resolve_run_dir(base_dir::AbstractString, config_path::AbstractString)
    run_id = run_id_from_config(config_path)
    config = effective_config(config_path)
    k = 1
    dir = joinpath(base_dir, run_id)
    while isdir(dir)
        snapshot = joinpath(dir, "config.toml")
        if isfile(snapshot) && effective_config(snapshot) == config
            if same_package_version(dir)
                meta = read_run_metadata(dir)
                finished =
                    haskey(meta, "finished") && get(meta, "failed_stages", "") == "" &&
                    isempty(abandoned_stages(dir))
                return dir, finished ? :complete : :resume
            end
            @warn "Skipping a sibling that matches the configuration: " *
                  version_mismatch_message(dir)
        end
        k += 1
        dir = joinpath(base_dir, "$(run_id)_r$k")
    end
    return unique_run_dir(base_dir, run_id), :fresh
end

"""
$(TYPEDSIGNATURES)

Write the configuration into the run directory as `config.toml`, so every
result set carries the exact configuration that produced it. A plain file
is copied verbatim; an overlay (`base_config`) is written as its merged
[`effective_config`](@ref), so the snapshot is self-contained and rehashes
to the run's identifier. Never overwrites an existing snapshot.
"""
function snapshot_config(config_path::AbstractString, run_dir::AbstractString)
    dest = joinpath(run_dir, "config.toml")
    isfile(dest) && throw(ArgumentError("configuration snapshot already exists: $dest"))
    if haskey(TOML.parsefile(config_path), BASE_CONFIG_KEY)
        open(dest, "w") do io
            TOML.print(io, effective_config(config_path); sorted = true)
        end
    else
        cp(config_path, dest; force = false)
    end
    return dest
end

"""
$(TYPEDSIGNATURES)

Copy the resolved manifest of the active project into the run directory as
`Manifest.toml`, so the exact dependency versions behind a result travel with
it (the repository tracks `Project.toml` and `[compat]` only). The
version-specific manifest names take precedence, as in the package loader.
Returns the destination, or `nothing` with a warning when the active project
has no manifest. Never overwrites an existing snapshot.
"""
function snapshot_manifest(run_dir::AbstractString)
    dest = joinpath(run_dir, "Manifest.toml")
    isfile(dest) && throw(ArgumentError("manifest snapshot already exists: $dest"))
    source = active_manifest()
    source === nothing || return cp(source, dest)
    @warn "No manifest found for the active project; the run directory carries no " *
          "dependency snapshot." project = Base.active_project()
    return nothing
end

# manifest of the active project, version-specific names first as in the
# package loader; `nothing` when there is none
function active_manifest()
    project = Base.active_project()
    project === nothing && return nothing
    suffix = "-v$(VERSION.major).$(VERSION.minor).toml"
    for name in ("JuliaManifest" * suffix, "Manifest" * suffix,
        "JuliaManifest.toml", "Manifest.toml")
        source = joinpath(dirname(project), name)
        isfile(source) && return source
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

`safesave` semantics for arbitrary file formats: if `path` exists, move the
existing file to `<name>#<k><ext>` with the smallest free `k` before the
caller writes the new file. Returns `path`.
"""
function backup_existing!(path::AbstractString)
    if isfile(path)
        stem, ext = splitext(path)
        k = 1
        while isfile("$(stem)#$(k)$(ext)")
            k += 1
        end
        mv(path, "$(stem)#$(k)$(ext)")
    end
    return path
end

"""
$(TYPEDSIGNATURES)

`git describe`-style identifier of the repository state (commit, tags,
dirty flag) via DrWatson, or `"unknown"` outside a repository.
"""
function git_state(project_root::AbstractString)
    desc = try
        gitdescribe(project_root)
    catch err
        # outside a repository (containers, tarballs) DrWatson throws; the
        # metadata then records "unknown" rather than aborting the run
        @debug "git state unavailable" project_root exception = err
        nothing
    end
    return desc === nothing ? "unknown" : desc
end

"""
$(TYPEDSIGNATURES)

Contents of `metadata.toml` in the run directory (empty when absent).
"""
function read_run_metadata(run_dir::AbstractString)
    path = joinpath(run_dir, "metadata.toml")
    return isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
end

"""
$(TYPEDSIGNATURES)

Write (or update) `metadata.toml` in the run directory with provenance
information passed as keyword pairs (git state, Julia version, backend,
threads, timings, …). Numbers, booleans, strings and string vectors are stored
natively, anything else through `string`. The file is replaced by a rename, so
an interrupted update leaves the previous record intact.
"""
function write_run_metadata(run_dir::AbstractString; kwargs...)
    path = joinpath(run_dir, "metadata.toml")
    meta = read_run_metadata(run_dir)
    for (k, v) in kwargs
        meta[String(k)] = v isa Union{Real,Bool,String,Vector{String}} ? v : string(v)
    end
    return replace_metadata!(path, meta)
end

# write `meta` to `path` through a temporary file and a rename, so an
# interrupted update leaves the previous record intact
function replace_metadata!(path::AbstractString, meta::AbstractDict)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        TOML.print(io, meta; sorted = true)
    end
    mv(tmp, path; force = true)
    return path
end

"""
$(TYPEDSIGNATURES)

Record the environment of attempt `attempt` of a run under
`[attempt_records.<attempt>]` in `metadata.toml`: start time, host, Julia
version, package version, git state, backend and the SHA-256 of the active
manifest. When the host, the Julia version or the manifest differs from the
first recorded attempt, a warning names the field: a continued run then mixes
results computed in two environments.
"""
function record_attempt!(run_dir::AbstractString, attempt::Integer; git::AbstractString,
    backend::AbstractString)
    meta = read_run_metadata(run_dir)
    records = get!(meta, "attempt_records", Dict{String,Any}())
    manifest = active_manifest()
    sha = manifest === nothing ? "" : bytes2hex(open(sha256, manifest))
    record = Dict{String,Any}("started" => string(now()), "hostname" => gethostname(),
        "julia_version" => string(VERSION), "package_version" => package_version(),
        "git" => String(git), "backend" => String(backend), "manifest_sha256" => sha)
    records[string(attempt)] = record
    first_record = get(records, "1", nothing)
    if attempt > 1 && first_record isa AbstractDict
        for field in ("hostname", "julia_version", "manifest_sha256")
            previous = get(first_record, field, nothing)
            previous == record[field] && continue
            @warn "Attempt $attempt of $run_dir runs with $field = $(record[field]), " *
                  "attempt 1 ran with $previous: the continued run mixes results " *
                  "computed in two environments."
        end
    end
    return replace_metadata!(joinpath(run_dir, "metadata.toml"), meta)
end

"""
$(TYPEDSIGNATURES)

Take the lock `name` in the run directory `dir`: a pid file that excludes a
second process from the same directory and is released by `close` on the
returned lock. A lock whose holder ran on this host and no longer exists (a
worker killed by the supervisor) is reclaimed under the guard lock
`<name>.reclaim`. An `ArgumentError` names the lock file when another live
process holds it, or a process of another host, whose lock is never reclaimed
and is removed by hand once that process has ended. A reclaim interrupted by a kill
leaves the guard behind, and the error then names both files, to be removed by
hand once no process works on `dir`.
"""
function acquire_run_lock(dir::AbstractString; name::AbstractString = RUN_LOCK)
    path = joinpath(dir, name)
    lock = try_pidlock(path)
    lock === nothing || return lock
    holder = lock_holder(path)
    # the pid lock of the standard library does not reclaim the file of a holder
    # that was killed (SIGKILL by the supervisor): reclaim it here when the holder
    # ran on this host and no longer exists
    if stale_lock(holder)
        lock = reclaim_stale_lock(path, holder)
        lock === nothing || return lock
        guard = path * ".reclaim"
        guard_holder = lock_holder(guard)
        stale_lock(guard_holder) && throw(
            ArgumentError(
                "the lock $path names a process that no longer exists ($holder), " *
                "and its reclaim was interrupted: the guard $guard names another " *
                "($guard_holder). Remove both files once no process works on $dir.",
            ),
        )
        holder = lock_holder(path)
    end
    fields = split(holder)
    remote = length(fields) == 2 && fields[2] != gethostname()
    throw(
        ArgumentError(
            "$dir is in use by another process (pid and host: $holder, lock file " *
            "$path); wait for it to finish or end it before continuing this run" *
            (
                remote ?
                ". A process of another host cannot be checked from here: if it no " *
                "longer runs (a job moved to another node), remove the lock file" : ""
            ),
        ),
    )
end

# Remove the lock file `path` of a holder that no longer exists, whose record
# read `holder`, and take the lock. The removal runs under a second lock, so
# that two processes finding the same stale file cannot both remove it: the
# one that gets the guard removes the file only if it still holds the stale
# record. Returns `nothing` when the guard is held elsewhere or the lock was
# taken in between.
function reclaim_stale_lock(path::String, holder::AbstractString)
    guard = try_pidlock(path * ".reclaim")
    guard === nothing && return nothing
    try
        lock_holder(path) == holder && rm(path; force = true)
        return try_pidlock(path)
    finally
        close(guard)
    end
end

# the pid lock at `path`, or `nothing` when another holder has it
function try_pidlock(path::AbstractString)
    try
        return mkpidlock(path; wait = false)
    catch err
        err isa PidlockedError || rethrow()
        return nothing
    end
end

# the record "<pid> <host>" of the lock file at `path`; "unknown" when the file
# does not exist, also when its holder removes it while it is read
function lock_holder(path::AbstractString)
    try
        return String(strip(read(path, String)))
    catch err
        err isa SystemError || rethrow()
        return "unknown"
    end
end

# true when a pid-lock record "<pid> <host>" names a process of this host that no
# longer exists; a record of another host is never stale here (no way to check)
function stale_lock(holder::AbstractString)
    fields = split(holder)
    length(fields) == 2 || return false
    pid = tryparse(Int32, fields[1])
    (pid === nothing || fields[2] != gethostname()) && return false
    return ccall(:uv_kill, Cint, (Cint, Cint), pid, 0) == Base.UV_ESRCH
end

"""
$(TYPEDSIGNATURES)

Key of a pipeline stage in the run records: `"sweep:<name>"` or `"map:<name>"`.
"""
stage_key(kind::Symbol, name::AbstractString) = string(kind, ':', name)

"""
$(TYPEDSIGNATURES)

Stages recorded as complete in `metadata.toml` ([`mark_stage_complete!`](@ref)).
"""
completed_stages(run_dir::AbstractString) =
    String.(get(read_run_metadata(run_dir), "completed_stages", String[]))

"""
$(TYPEDSIGNATURES)

Record stage `key` as complete. Called after the stage's outputs are written,
so a stage interrupted during persistence is redone on the next attempt.
"""
function mark_stage_complete!(run_dir::AbstractString, key::AbstractString)
    done = completed_stages(run_dir)
    key in done || push!(done, String(key))
    write_run_metadata(run_dir; completed_stages = done)
    return done
end

"""
$(TYPEDSIGNATURES)

Record in `metadata.toml` (`figure_failures`) that the figures of stage `key`
could not be rendered; the numerical outputs of the stage are complete and the
figures can be rebuilt from them.
"""
function note_figure_failure!(run_dir::AbstractString, key::AbstractString)
    failed = String.(get(read_run_metadata(run_dir), "figure_failures", String[]))
    key in failed || push!(failed, String(key))
    write_run_metadata(run_dir; figure_failures = failed)
    return failed
end

"""
$(TYPEDSIGNATURES)

Stages a supervisor gave up on (`abandoned` in the run's `supervision.toml`);
a resumed run skips them.
"""
function abandoned_stages(run_dir::AbstractString)
    path = joinpath(run_dir, SUPERVISION_LEDGER)
    isfile(path) || return String[]
    return String.(get(TOML.parsefile(path), "abandoned", String[]))
end

"""
$(TYPEDSIGNATURES)

Write the platform fingerprint sidecar `hardware.txt` into the run
directory: Julia's own `versioninfo` report (Julia/OS/CPU/threads), host
totals not covered by it (logical CPU threads, total memory, BLAS
threads), and, when a GPU backend is active, the backend runtime's device
fingerprint (`device_fingerprint`: driver/runtime versions and device
inventory). Together with `metadata.toml` (config hash, git state) this
makes every result attributable to config + commit + hardware.
"""
function write_hardware_fingerprint(run_dir::AbstractString;
    device_report::AbstractString = "")
    path = joinpath(run_dir, "hardware.txt")
    open(path, "w") do io
        versioninfo(io)
        println(io)
        println(io, "Logical CPU threads: ", Sys.CPU_THREADS)
        println(io, "Total memory: ",
            round(Sys.total_memory() / 2^30, digits = 1), " GiB")
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        if !isempty(device_report)
            println(io)
            println(io, "── GPU backend ──")
            print(io, device_report)
        end
    end
    return path
end

end # module
