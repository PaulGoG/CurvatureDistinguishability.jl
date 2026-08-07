"""
Run identity and safekeeping: canonical configuration hashing, config
snapshots, backup-before-overwrite semantics and run metadata.
"""
module Provenance

using SHA: sha256
using TOML: TOML
using DrWatson: gitdescribe

export run_id_from_config,
    unique_run_dir, snapshot_config, backup_existing!,
    write_run_metadata, git_state

"""
    run_id_from_config(config_path) -> String

Deterministic run identifier: the first 8 hex characters of the SHA-256 of
the canonically serialized *parsed* configuration (keys sorted, values
only — never the wall clock), so identical physical/numerical content maps
to identical IDs and reruns are recognizable. Comments and formatting do
not affect a run's identity.
"""
function run_id_from_config(config_path::AbstractString)
    canonical = sprint(io -> TOML.print(io, TOML.parsefile(config_path); sorted = true))
    return "run_" * first(bytes2hex(sha256(canonical)), 8)
end

"""
    unique_run_dir(base_dir, run_id) -> String

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
    snapshot_config(config_path, run_dir)

Copy the configuration file into the run directory so every result set
carries the exact configuration that produced it.
"""
snapshot_config(config_path::AbstractString, run_dir::AbstractString) =
    cp(config_path, joinpath(run_dir, "config.toml"); force = false)

"""
    backup_existing!(path)

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
    git_state(project_root) -> String

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
    write_run_metadata(run_dir; kwargs...)

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

end # module
