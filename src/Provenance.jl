"""
Run identity and safekeeping: canonical configuration hashing, config
snapshots, backup-before-overwrite semantics and run metadata.
"""
module Provenance

using DocStringExtensions: TYPEDSIGNATURES
using SHA: sha256
using TOML: TOML
using DrWatson: gitdescribe
using InteractiveUtils: versioninfo
using LinearAlgebra: BLAS

export run_id_from_config, effective_config,
    unique_run_dir, snapshot_config, backup_existing!,
    write_run_metadata, write_hardware_fingerprint, git_state

# top-level key naming the base file an overlay configuration is merged onto
const BASE_CONFIG_KEY = "base_config"

"""
$(TYPEDSIGNATURES)

Parsed configuration with its base file resolved. A top-level
`base_config = "file.toml"` (path relative to the overlay's directory)
names a base whose tables are deep-merged beneath the overlay's: sub-tables
recurse, while scalars, arrays and arrays of tables (`[[sweeps]]`,
`[[maps]]`) present in the overlay replace the base's. One level only — a
base that itself declares `base_config` is an error, as is a missing base.
The `base_config` key is stripped, so the result is the self-contained
table every consumer (validation, run-ID hashing, snapshots) operates on; a
file without `base_config` parses as-is.
"""
function effective_config(config_path::AbstractString)
    config = TOML.parsefile(config_path)
    haskey(config, BASE_CONFIG_KEY) || return config
    base_rel = pop!(config, BASE_CONFIG_KEY)
    base_rel isa AbstractString || error(
        "$BASE_CONFIG_KEY in $config_path must be a file path, got $(repr(base_rel))",
    )
    base_path = normpath(joinpath(dirname(abspath(config_path)), base_rel))
    isfile(base_path) ||
        error("$BASE_CONFIG_KEY of $config_path names a missing file: $base_path")
    base = TOML.parsefile(base_path)
    haskey(base, BASE_CONFIG_KEY) && error(
        "base configuration $base_path declares $BASE_CONFIG_KEY itself; " *
        "only one overlay level is supported",
    )
    return merge_config(base, config)
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
the canonically serialized *effective* configuration
([`effective_config`](@ref): keys sorted, values only — never the wall
clock), so identical physical/numerical content maps to identical IDs and
reruns are recognizable. Comments, formatting and the split between a base
file and its overlay do not affect a run's identity.
"""
function run_id_from_config(config_path::AbstractString)
    canonical = sprint(io -> TOML.print(io, effective_config(config_path); sorted = true))
    return "run_" * first(bytes2hex(sha256(canonical)), 8)
end

"""
$(TYPEDSIGNATURES)

Create and return an output directory for `run_id` under `base_dir`. If the
directory already exists (a rerun of the same configuration), a `_r2`,
`_r3`, … suffix is appended — existing results are never overwritten.
"""
function unique_run_dir(base_dir::AbstractString, run_id::AbstractString)
    dir = joinpath(base_dir, run_id)
    k = 1
    while isdir(dir)
        k += 1
        dir = joinpath(base_dir, "$(run_id)_r$k")
    end
    mkpath(dir)
    return dir
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
    isfile(dest) && error("configuration snapshot already exists: $dest")
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
    catch
        nothing
    end
    return desc === nothing ? "unknown" : desc
end

"""
$(TYPEDSIGNATURES)

Write (or update) `metadata.toml` in the run directory with provenance
information passed as keyword pairs (git state, Julia version, backend,
threads, timings, …). Values are stored as strings for TOML robustness.
"""
function write_run_metadata(run_dir::AbstractString; kwargs...)
    path = joinpath(run_dir, "metadata.toml")
    meta = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
    for (k, v) in kwargs
        meta[String(k)] = v isa Union{Real,Bool,String} ? v : string(v)
    end
    open(path, "w") do io
        TOML.print(io, meta)
    end
    return path
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
