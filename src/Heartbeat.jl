"""
Liveness telemetry of a worker process: a monotone tick counter advanced at
every loss evaluation and finished work item, and a background writer that
publishes it to a file a supervising process can poll (`Supervision`). A tick
is one atomic increment; nothing is written unless a writer was started.
"""
module Heartbeat

using DocStringExtensions: TYPEDSIGNATURES
using Dates: now
using TOML: TOML

export tick!, note_item!, start_heartbeat, stop_heartbeat!, read_heartbeat

const TICKS = Threads.Atomic{Int}(0)
const ITEMS = Threads.Atomic{Int}(0)

"""
Advance the tick counter: called once per loss evaluation (every kernel launch
on a GPU backend), so the interval between ticks is bounded by the duration of
one launch whatever the duration of a work item.
"""
@inline function tick!()
    Threads.atomic_add!(TICKS, 1)
    return nothing
end

"""
Record one finished work item (a fitted separation, a mapped direction) and
tick.
"""
function note_item!()
    Threads.atomic_add!(ITEMS, 1)
    return tick!()
end

"""
Handle of a running heartbeat writer ([`start_heartbeat`](@ref)).
"""
struct HeartbeatWriter
    path::String
    running::Threads.Atomic{Bool}
    task::Task
end

"""
$(TYPEDSIGNATURES)

Write the current counters to `path` through a temporary file and a rename, so
a concurrent reader never sees a partial file.
"""
function write_heartbeat(path::AbstractString)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        TOML.print(io,
            Dict{String,Any}("pid" => Int(getpid()), "ticks" => TICKS[],
                "items_done" => ITEMS[], "time" => string(now())))
    end
    mv(tmp, path; force = true)
    return path
end

"""
$(TYPEDSIGNATURES)

Publish the counters to `path` every `period_s` seconds until
[`stop_heartbeat!`](@ref). The writer runs on the interactive thread pool when
one exists, so compute tasks that do not yield cannot starve it.
"""
function start_heartbeat(path::AbstractString, period_s::Real)
    period_s > 0 || throw(ArgumentError("heartbeat period must be > 0, got $period_s"))
    running = Threads.Atomic{Bool}(true)
    write_heartbeat(path)
    task = Threads.@spawn :interactive begin
        while running[]
            sleep(period_s)
            write_heartbeat(path)
        end
    end
    return HeartbeatWriter(String(path), running, task)
end

"""
$(TYPEDSIGNATURES)

Stop the writer after a final write.
"""
function stop_heartbeat!(writer::HeartbeatWriter)
    writer.running[] = false
    wait(writer.task)
    write_heartbeat(writer.path)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Counters last published to `path` as `(ticks, items_done)`, or `nothing` while
no heartbeat exists.
"""
function read_heartbeat(path::AbstractString)
    isfile(path) || return nothing
    table = TOML.tryparsefile(path)
    table isa TOML.ParserError && return nothing
    return (
        ticks = Int(get(table, "ticks", 0)),
        items_done = Int(get(table, "items_done", 0)),
    )
end

end # module
