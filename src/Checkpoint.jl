"""
Work-item checkpoint of a sweep: one durable row per fitted separation, so an
interrupted sweep continues from its finished items. Rows carry the exact
`Float64` values (shortest round-trip representation); a resumed sweep
therefore reproduces the table of an uninterrupted one.
"""
module Checkpoint

using DocStringExtensions: TYPEDSIGNATURES

export CheckpointRow, SweepCheckpoint, open_checkpoint, record_item!, count_items

const FORMAT_LINE = "# sweep checkpoint v1"
const COLUMNS =
    "index,D2,best_fit,converged,iterations,g_norm,at_bound," *
    "multi_start_gain,wall_seconds,attempt"

"""
Result of one work item: the separation index, its squared distance and
best-fit parameters, the optimizer diagnostics, the wall time of the item and
the attempt (process launch) that produced it.
"""
struct CheckpointRow
    index::Int
    D2::Float64
    best_fit::Vector{Float64}
    converged::Bool
    iterations::Int
    g_norm::Float64
    at_bound::Bool
    multi_start_gain::Float64
    wall_seconds::Float64
    attempt::Int
end

"""
An open checkpoint file: `rows` holds the items found at opening time, keyed by
separation index; [`record_item!`](@ref) appends further ones.
"""
struct SweepCheckpoint
    path::String
    rows::Dict{Int,CheckpointRow}
    lock::ReentrantLock
end

encode(row::CheckpointRow) = join(
    (row.index, repr(row.D2), join(repr.(row.best_fit), ';'), row.converged,
        row.iterations, repr(row.g_norm), row.at_bound, repr(row.multi_start_gain),
        repr(row.wall_seconds), row.attempt), ',')

# `nothing` for a line that is not a complete row (a write cut short by a kill)
function decode(line::AbstractString)
    fields = split(line, ',')
    length(fields) == 10 || return nothing
    index = tryparse(Int, fields[1])
    D2 = tryparse(Float64, fields[2])
    best_fit = tryparse.(Float64, split(fields[3], ';'))
    converged = tryparse(Bool, fields[4])
    iterations = tryparse(Int, fields[5])
    g_norm = tryparse(Float64, fields[6])
    at_bound = tryparse(Bool, fields[7])
    gain = tryparse(Float64, fields[8])
    wall = tryparse(Float64, fields[9])
    attempt = tryparse(Int, fields[10])
    any(isnothing,
        (index, D2, converged, iterations, g_norm, at_bound, gain, wall, attempt)) &&
        return nothing
    any(isnothing, best_fit) && return nothing
    return CheckpointRow(index, D2, Float64.(best_fit), converged, iterations, g_norm,
        at_bound, gain, wall, attempt)
end

# flush Julia's buffer and the kernel's: a host reset must not lose finished items
function sync_to_disk(io::IOStream)
    flush(io)
    if Sys.isunix()
        raw = fd(io)
        handle = raw isa RawFD ? reinterpret(Int32, raw) : Int32(raw)
        ccall(:fsync, Cint, (Cint,), handle) == 0 ||
            throw(SystemError("fsync of the sweep checkpoint"))
    end
    return nothing
end

function write_all(path::AbstractString, signature::AbstractString, rows)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, FORMAT_LINE, " signature=", signature)
        println(io, COLUMNS)
        foreach(row -> println(io, encode(row)), rows)
        sync_to_disk(io)
    end
    mv(tmp, path; force = true)
    return path
end

"""
$(TYPEDSIGNATURES)

Open the checkpoint at `path`, creating it when absent or empty. `signature`
identifies the problem the rows belong to (grid, direction, base point,
optimizer settings, seed); a file written under another signature is refused
with an `ArgumentError`. An incomplete last line — a write cut short when the
process was killed — is dropped; an unreadable line anywhere else is an
error.
"""
function open_checkpoint(path::AbstractString, signature::AbstractString)
    rows = Dict{Int,CheckpointRow}()
    lines = isfile(path) ? readlines(path) : String[]
    if isempty(lines)
        write_all(path, signature, CheckpointRow[])
        return SweepCheckpoint(String(path), rows, ReentrantLock())
    end
    lines[1] == "$FORMAT_LINE signature=$signature" || throw(
        ArgumentError(
            "checkpoint $path was written for a different problem or format " *
            "(header '$(lines[1])'); move it aside to recompute the sweep",
        ),
    )
    body = lines[3:end]
    truncated = false
    for (k, line) in enumerate(body)
        row = decode(line)
        if row === nothing
            k == length(body) || throw(
                ArgumentError("checkpoint $path: unreadable row $(k + 2): '$line'"),
            )
            truncated = true
        else
            rows[row.index] = row
        end
    end
    truncated && write_all(path, signature, sort!(collect(values(rows)); by = r -> r.index))
    return SweepCheckpoint(String(path), rows, ReentrantLock())
end

"""
$(TYPEDSIGNATURES)

Append `row` durably (written, flushed and synced before returning). Safe to
call from concurrent tasks.
"""
function record_item!(checkpoint::SweepCheckpoint, row::CheckpointRow)
    lock(checkpoint.lock) do
        open(checkpoint.path, "a") do io
            println(io, encode(row))
            sync_to_disk(io)
        end
        checkpoint.rows[row.index] = row
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Number of complete rows in the checkpoint file at `path` (0 when absent), read
without validating its signature.
"""
function count_items(path::AbstractString)
    isfile(path) || return 0
    return count(line -> decode(line) !== nothing, Iterators.drop(eachline(path), 2))
end

end # module
