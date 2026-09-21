"""
Supervision of a worker process from a second process that never touches the
worker's devices: hang detection from the conjunction of stalled progress and
idle CPU, termination with signal escalation, and relaunch under a retry
budget. A lost GPU dispatch, a reset device context or a crashed driver cannot
be recovered inside the affected process; they are recovered by replacing it.

The module is independent of the pipeline (standard library only): a
[`WorkerSpec`](@ref) describes how to launch the worker and how to observe its
progress.
"""
module Supervision

using DocStringExtensions: TYPEDSIGNATURES
using Dates: now
using TOML: TOML
using ..Heartbeat: read_heartbeat

export SupervisionSettings, WorkerSpec, supervise
public cpu_seconds, AttemptRecord, SupervisionResult, EXIT_CONFIG_ERROR

"""
Exit status by which a worker reports an input it cannot honour; such an
attempt is never retried.
"""
const EXIT_CONFIG_ERROR = 2

# sysconf(_SC_CLK_TCK) selector on Linux
const SC_CLK_TCK = 2

"""
    SupervisionSettings(; kwargs...)

Thresholds and budget of [`supervise`](@ref) (the `[supervision]` table of a
configuration). Times in seconds.

  - `max_retries = 5`: relaunches after the first attempt, ≥ 0.
  - `max_stalled_retries = 2`: consecutive attempts that finish no new work
    item before the blocking unit is abandoned, ≥ 1.
  - `poll_interval_s = 15`: sampling period of progress and CPU time, in [0.05, 600].
  - `startup_grace_s = 1800`: deadline for the worker's first heartbeat, > 0.
  - `idle_stall_s = 300`: no progress with the CPU idle for this long is a
    hang, ≥ 4 `poll_interval_s`.
  - `busy_stall_s = 7200`: no progress with the CPU busy for this long ends the
    attempt, ≥ `idle_stall_s`.
  - `cpu_idle_cores = 0.05`: utilisation below which the worker is idle, in (0, 1].
  - `term_grace_s = 60`: wait between `SIGTERM` and `SIGKILL`, > 0.
  - `kill_wait_s = 120`: wait for the process to disappear after `SIGKILL`, > 0.
  - `backoff_s = 30`: pause before a relaunch, ≥ 0.

Throws `ArgumentError` naming the offending key.
"""
struct SupervisionSettings
    max_retries::Int
    max_stalled_retries::Int
    poll_interval_s::Float64
    startup_grace_s::Float64
    idle_stall_s::Float64
    busy_stall_s::Float64
    cpu_idle_cores::Float64
    term_grace_s::Float64
    kill_wait_s::Float64
    backoff_s::Float64

    function SupervisionSettings(; max_retries::Integer = 5,
        max_stalled_retries::Integer = 2, poll_interval_s::Real = 15.0,
        startup_grace_s::Real = 1800.0, idle_stall_s::Real = 300.0,
        busy_stall_s::Real = 7200.0, cpu_idle_cores::Real = 0.05,
        term_grace_s::Real = 60.0, kill_wait_s::Real = 120.0, backoff_s::Real = 30.0)
        bad(key, rule, value) =
            throw(ArgumentError("[supervision].$key must be $rule, got $value"))
        max_retries >= 0 || bad("max_retries", ">= 0", max_retries)
        max_stalled_retries >= 1 || bad("max_stalled_retries", ">= 1", max_stalled_retries)
        0.05 <= poll_interval_s <= 600 ||
            bad("poll_interval_s", "in [0.05, 600]", poll_interval_s)
        startup_grace_s > 0 || bad("startup_grace_s", "> 0", startup_grace_s)
        idle_stall_s >= 4 * poll_interval_s ||
            bad("idle_stall_s", ">= 4 poll_interval_s", idle_stall_s)
        busy_stall_s >= idle_stall_s || bad("busy_stall_s", ">= idle_stall_s", busy_stall_s)
        0 < cpu_idle_cores <= 1 || bad("cpu_idle_cores", "in (0, 1]", cpu_idle_cores)
        term_grace_s > 0 || bad("term_grace_s", "> 0", term_grace_s)
        kill_wait_s > 0 || bad("kill_wait_s", "> 0", kill_wait_s)
        backoff_s >= 0 || bad("backoff_s", ">= 0", backoff_s)
        return new(max_retries, max_stalled_retries, poll_interval_s, startup_grace_s,
            idle_stall_s, busy_stall_s, cpu_idle_cores, term_grace_s, kill_wait_s,
            backoff_s)
    end
end

"""
    WorkerSpec(; command, console_log, heartbeat_path, progress_paths, items_done,
               status, abandon_stalled!)

What [`supervise`](@ref) needs to know about a worker:

  - `command(attempt) -> Cmd`: the process to launch for attempt number `attempt`.
  - `console_log(attempt) -> String`: file receiving its stdout and stderr.
  - `heartbeat_path`: file the worker publishes through `Heartbeat`.
  - `progress_paths() -> Vector{String}`: files whose growth counts as progress
    beside the heartbeat ticks.
  - `items_done() -> Int`: finished work items so far, monotone across attempts
    (an attempt that raises it was productive).
  - `status() -> Symbol`: `:complete`, `:partial` (nothing left to attempt, some
    units abandoned) or `:incomplete`, verified from the outputs.
  - `abandon_stalled!() -> Union{Nothing,String}`: give up on the unit that keeps
    failing without progress and return its name, or `nothing` when nothing can
    be abandoned.
"""
struct WorkerSpec{C,L,P,I,S,A}
    command::C
    console_log::L
    heartbeat_path::String
    progress_paths::P
    items_done::I
    status::S
    abandon_stalled!::A
end

function WorkerSpec(; command, console_log, heartbeat_path::AbstractString,
    progress_paths = () -> String[], items_done = () -> 0,
    status = () -> :complete, abandon_stalled! = () -> nothing)
    return WorkerSpec(command, console_log, String(heartbeat_path), progress_paths,
        items_done, status, abandon_stalled!)
end

"""
One attempt as recorded in the ledger. `verdict` is one of `:complete`,
`:partial`, `:incomplete` (exit status 0 without verified outputs), `:failed`,
`:crashed` (ended by a signal it was not sent by the supervisor),
`:config_error`, `:stalled_start`, `:hang`, `:livelock`, `:wedged` (survived
`SIGKILL`).
"""
struct AttemptRecord
    attempt::Int
    started::String
    finished::String
    verdict::Symbol
    exit_code::Int
    signal::Int
    elapsed_s::Float64
    cpu_s::Float64
    stall_s::Float64
    items_before::Int
    items_after::Int
end

"""
Outcome of [`supervise`](@ref): `status` is `:complete`, `:partial`,
`:config_error`, `:wedged`, `:stalled` (no progress and nothing left to
abandon) or `:retries_exhausted`.
"""
struct SupervisionResult
    status::Symbol
    attempts::Vector{AttemptRecord}
    abandoned::Vector{String}
end

"""
$(TYPEDSIGNATURES)

CPU seconds consumed so far by process `pid` and its reaped children (user +
system), from `/proc/<pid>/stat`; `nothing` where `/proc` is unavailable or
the process is gone. Fields are counted after the closing parenthesis of the
command name, which may itself contain spaces and parentheses.
"""
function cpu_seconds(pid::Integer)
    path = "/proc/$pid/stat"
    isfile(path) || return nothing
    raw = try
        read(path, String)
    catch err
        # the process can exit between the check and the read
        err isa Union{SystemError,Base.IOError} || rethrow()
        return nothing
    end
    close_paren = findlast(')', raw)
    close_paren === nothing && return nothing
    fields = split(raw[(close_paren+1):end])
    length(fields) >= 15 || return nothing
    # after the command name: state, ppid, …, utime (12), stime (13),
    # cutime (14), cstime (15)
    jiffies = sum(parse(Int, fields[i]) for i in 12:15)
    return jiffies / ccall(:sysconf, Clong, (Cint,), SC_CLK_TCK)
end

# progress observable: heartbeat ticks and the sizes of the watched files
function observe(spec::WorkerSpec)
    beat = read_heartbeat(spec.heartbeat_path)
    bytes = 0
    for path in spec.progress_paths()
        isfile(path) && (bytes += filesize(path))
    end
    return (ticks = beat === nothing ? -1 : beat.ticks, bytes = bytes)
end

# CPU utilisation in cores over the trailing window, or `nothing` until the
# samples span 80 % of it
function trailing_utilisation(samples::Vector{Tuple{Float64,Float64}}, window_s::Real)
    isempty(samples) && return nothing
    t_now, cpu_now = samples[end]
    while length(samples) > 1 && t_now - samples[2][1] >= window_s
        popfirst!(samples)
    end
    t_old, cpu_old = samples[1]
    t_now - t_old >= 0.8 * window_s || return nothing
    return (cpu_now - cpu_old) / (t_now - t_old)
end

wait_exit(proc, timeout_s) =
    timedwait(() -> process_exited(proc), Float64(timeout_s);
        pollint = 0.05) === :ok

"""
$(TYPEDSIGNATURES)

End `proc`: `SIGTERM`, then `SIGKILL` after `term_grace_s`. Returns `false`
when the process survives `kill_wait_s` after `SIGKILL` (uninterruptible
device wait): nothing further can be launched onto that device.
"""
function terminate!(proc::Base.Process, s::SupervisionSettings)
    process_exited(proc) && return true
    kill(proc, Base.SIGTERM)
    wait_exit(proc, s.term_grace_s) && return true
    kill(proc, Base.SIGKILL)
    return wait_exit(proc, s.kill_wait_s)
end

function run_attempt(spec::WorkerSpec, s::SupervisionSettings, attempt::Int)
    console = spec.console_log(attempt)
    mkpath(dirname(console))
    rm(spec.heartbeat_path; force = true)
    items_before = spec.items_done()
    started = string(now())
    t_start = time()
    proc = open(console, "a") do io
        run(pipeline(spec.command(attempt); stdout = io, stderr = io); wait = false)
    end
    pid = getpid(proc)
    @info "Attempt $attempt: worker pid $pid, console log $console"

    verdict = :none
    last_progress = observe(spec)
    t_progress = t_start
    cpu_last = 0.0
    samples = Tuple{Float64,Float64}[]
    cpu_available = cpu_seconds(pid) !== nothing
    cpu_available ||
        @warn "CPU time of the worker is not observable on this platform; only the " *
              "busy_stall_s deadline applies."
    try
        while !wait_exit(proc, s.poll_interval_s)
            t = time()
            observed = observe(spec)
            if observed != last_progress
                last_progress = observed
                t_progress = t
            end
            cpu = cpu_seconds(pid)
            if cpu !== nothing
                cpu_last = cpu
                push!(samples, (t, cpu))
            end
            stall = t - t_progress
            if observed.ticks < 0
                # no heartbeat yet: environment instantiation, precompilation
                t - t_start > s.startup_grace_s && (verdict = :stalled_start)
            elseif stall >= s.busy_stall_s
                verdict = :livelock
            elseif stall >= s.idle_stall_s && cpu !== nothing
                u = trailing_utilisation(samples, s.idle_stall_s)
                u !== nothing && u < s.cpu_idle_cores && (verdict = :hang)
            end
            if verdict !== :none
                @warn "Attempt $attempt: $(verdict) — no progress for " *
                      "$(round(Int, stall)) s; terminating pid $pid."
                terminate!(proc, s) || (verdict = :wedged)
                break
            end
        end
    catch
        # the supervisor itself is being stopped: do not leave the worker behind
        terminate!(proc, s)
        rethrow()
    end

    stall_s = time() - t_progress
    if verdict === :none
        verdict =
            proc.termsignal != 0 ? :crashed :
            proc.exitcode == EXIT_CONFIG_ERROR ? :config_error :
            proc.exitcode != 0 ? :failed : spec.status()
        stall_s = 0.0
    end
    return AttemptRecord(attempt, started, string(now()), verdict,
        verdict === :wedged ? -1 : proc.exitcode, verdict === :wedged ? 0 : proc.termsignal,
        time() - t_start, cpu_last, stall_s, items_before, spec.items_done())
end

function write_ledger(path::AbstractString, status::Symbol,
    attempts::Vector{AttemptRecord}, abandoned::Vector{String})
    table = Dict{String,Any}("status" => String(status), "abandoned" => abandoned,
        "attempts" => [
            Dict{String,Any}(
                String(f) => (v = getfield(a, f); v isa Symbol ? String(v) : v)
                for f in fieldnames(AttemptRecord)
            ) for a in attempts
        ])
    tmp = path * ".tmp"
    open(io -> TOML.print(io, table; sorted = true), tmp, "w")
    mv(tmp, path; force = true)
    return path
end

"""
$(TYPEDSIGNATURES)

Run the worker of `spec` to completion under the budget of `settings`,
recording every attempt in the TOML ledger `ledger_path` (rewritten atomically
after each attempt; `abandoned` lists the units given up on).

An attempt ends when the worker exits or when the watchdog ends it: no
heartbeat within `startup_grace_s`; no progress for `idle_stall_s` while the
CPU utilisation over that window is below `cpu_idle_cores` (a hang — the
process waits on something that never arrives); or no progress for
`busy_stall_s` whatever the CPU does. Progress is any change of the heartbeat
tick count or of the total size of `spec.progress_paths()`.

After an attempt that is neither complete nor a configuration error the worker
is relaunched, at most `max_retries` times. Attempts that finish no new work
item are counted; `max_stalled_retries` of them in a row abandon the blocking
unit (`spec.abandon_stalled!`), so a deterministic failure ends in a few
attempts while failures that arrive after some progress draw on `max_retries`
only.
"""
function supervise(spec::WorkerSpec, settings::SupervisionSettings;
    ledger_path::AbstractString)
    attempts = AttemptRecord[]
    abandoned = String[]
    function finish(status::Symbol)
        write_ledger(ledger_path, status, attempts, abandoned)
        return SupervisionResult(status, attempts, abandoned)
    end
    stalled = 0
    for attempt in 1:(1+settings.max_retries)
        record = run_attempt(spec, settings, attempt)
        push!(attempts, record)
        write_ledger(ledger_path, :running, attempts, abandoned)
        @info "Attempt $attempt ended: $(record.verdict) (exit $(record.exit_code), " *
              "signal $(record.signal), $(record.items_after - record.items_before) new items, " *
              "$(round(record.elapsed_s, digits = 1)) s)."
        record.verdict in (:complete, :partial, :config_error, :wedged) &&
            return finish(record.verdict)
        stalled = record.items_after > record.items_before ? 0 : stalled + 1
        if stalled >= settings.max_stalled_retries
            unit = spec.abandon_stalled!()
            unit === nothing && return finish(:stalled)
            @warn "No progress in $stalled consecutive attempts: abandoning '$unit'."
            push!(abandoned, unit)
            # the next worker reads the abandoned units from the ledger
            write_ledger(ledger_path, :running, attempts, abandoned)
            stalled = 0
        end
        attempt <= settings.max_retries && sleep(settings.backoff_s)
    end
    return finish(:retries_exhausted)
end

end # module
