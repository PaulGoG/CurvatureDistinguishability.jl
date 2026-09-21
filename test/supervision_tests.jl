# Process supervision, heartbeat, work-item checkpoints and run continuation.
# Fake workers are short Julia processes whose behaviour is fixed by a one-line
# program; thresholds are seconds, so every verdict is reached quickly.

const SUP = CD.Supervision
const CKPT = CD.Checkpoint

# plain binary and a scrubbed environment, as for the other child processes of
# the suite (Pkg.test exports its sandbox through JULIA_LOAD_PATH/JULIA_PROJECT)
const JULIA_BIN = Base.julia_cmd().exec[1]
const CLEAN_ENV = ["JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing]
fake_worker(program::AbstractString) =
    addenv(`$JULIA_BIN --startup-file=no --threads=1 -e $program`, CLEAN_ENV...)

fast_settings(; kwargs...) = SupervisionSettings(; poll_interval_s = 0.1,
    idle_stall_s = 1.0, busy_stall_s = 4.0, cpu_idle_cores = 0.2,
    startup_grace_s = 30.0,
    term_grace_s = 2.0, kill_wait_s = 10.0, backoff_s = 0.0, kwargs...)

# heartbeat file as the pipeline worker would publish it
beat_program(
    path,
    ticks,
) = "write($(repr(path)), \"ticks = $ticks\\nitems_done = 0\\npid = 0\\ntime = \\\"t\\\"\\n\")"

@testset "Supervision" begin
    @testset "settings are validated and name the key" begin
        @test SupervisionSettings().max_retries == 5
        @test_throws "[supervision].max_retries" SupervisionSettings(; max_retries = -1)
        @test_throws "[supervision].idle_stall_s" SupervisionSettings(;
            poll_interval_s = 10.0, idle_stall_s = 30.0)
        @test_throws "[supervision].busy_stall_s" SupervisionSettings(;
            idle_stall_s = 300.0, busy_stall_s = 100.0)
        @test_throws "[supervision].cpu_idle_cores" SupervisionSettings(;
            cpu_idle_cores = 0.0)
        mktempdir() do dir
            path = joinpath(dir, "c.toml")
            write(path, "[supervision]\nmax_retries = 9\nidle_stall_s = 120.0\n")
            cfg = load_and_validate_config(path)
            @test cfg.supervision.max_retries == 9
            @test cfg.supervision.idle_stall_s == 120.0
            write(path, "[supervision]\nmax_retrys = 9\n")
            @test_throws "max_retrys" load_and_validate_config(path)
            write(path, "[supervision]\nbackoff_s = -1.0\n")
            @test_throws "[supervision].backoff_s" load_and_validate_config(path)
            # execution section: it never enters the run identity
            write(path, "[supervision]\nmax_retries = 9\n")
            id_a = run_id_from_config(path)
            write(path, "[supervision]\nmax_retries = 1\n")
            @test run_id_from_config(path) == id_a
        end
    end

    if Sys.islinux()
        @testset "CPU time of a process" begin
            before = SUP.cpu_seconds(getpid())
            @test before isa Float64 && before > 0
            x = 0.0
            for i in 1:30_000_000
                x += sqrt(i)
            end
            @test SUP.cpu_seconds(getpid()) > before
            @test SUP.cpu_seconds(2^22 + 12345) === nothing # no such process
        end
    end

    @testset "heartbeat" begin
        mktempdir() do dir
            path = joinpath(dir, "heartbeat.toml")
            @test CD.Heartbeat.read_heartbeat(path) === nothing
            writer = CD.Heartbeat.start_heartbeat(path, 0.05)
            first_beat = CD.Heartbeat.read_heartbeat(path)
            CD.Heartbeat.tick!()
            CD.Heartbeat.note_item!()
            CD.Heartbeat.stop_heartbeat!(writer)
            last_beat = CD.Heartbeat.read_heartbeat(path)
            @test last_beat.ticks == first_beat.ticks + 2
            @test last_beat.items_done == first_beat.items_done + 1
            @test !isfile(path * ".tmp")
            @test_throws ArgumentError CD.Heartbeat.start_heartbeat(path, 0.0)
        end
    end

    @testset "sweep checkpoint" begin
        mktempdir() do dir
            path = joinpath(dir, "checkpoint.csv")
            ck = CKPT.open_checkpoint(path, "sig")
            @test isempty(ck.rows) && CKPT.count_items(path) == 0
            rows = [
                CKPT.CheckpointRow(3, 1.2345678901234567e-13,
                    [0.1, 1 / 3, -2.5e-300, 0.0, 5.0e-324, 0.7], true, 17, 3.0e-9,
                    false,
                    1.0, 12.5, 1),
                CKPT.CheckpointRow(1, Inf, [NaN, 1.0, 2.0, 3.0, 4.0, 5.0], false, 60,
                    NaN, true, 2.5, 0.25, 2),
            ]
            foreach(r -> CKPT.record_item!(ck, r), rows)
            again = CKPT.open_checkpoint(path, "sig")
            @test sort(collect(keys(again.rows))) == [1, 3]
            for r in rows, f in fieldnames(CKPT.CheckpointRow)
                @test isequal(getfield(again.rows[r.index], f), getfield(r, f)) # bit-exact
            end
            @test CKPT.count_items(path) == 2
            # a write cut short by a kill: the partial last line is dropped and
            # the file is left appendable
            open(io -> print(io, "2,1.0e-5,0.1;0.2"), path, "a")
            healed = CKPT.open_checkpoint(path, "sig")
            @test sort(collect(keys(healed.rows))) == [1, 3]
            @test all(line -> count(==(','), line) == 9, readlines(path)[3:end])
            # rows of another problem are refused; damage elsewhere is an error
            @test_throws ArgumentError CKPT.open_checkpoint(path, "other")
            lines = readlines(path)
            write(path, join([lines[1], lines[2], "garbage", lines[3]], '\n') * "\n")
            @test_throws ArgumentError CKPT.open_checkpoint(path, "sig")
            # an empty file (unflushed creation before a host reset) is a new checkpoint
            write(path, "")
            @test isempty(CKPT.open_checkpoint(path, "sig").rows)
        end
    end

    @testset "watchdog verdicts and retry budget" begin
        mktempdir() do dir
            beat = joinpath(dir, "heartbeat.toml")
            ledger = joinpath(dir, "supervision.toml")
            log_of(attempt) = joinpath(dir, "logs", "attempt_$attempt.log")

            # blocked: heartbeat once, then idle for ever — a hang, detected from
            # no progress AND idle CPU; never productive, nothing to abandon
            spec = WorkerSpec(;
                command = _ -> fake_worker(beat_program(beat, 1) * "; sleep(3600)"),
                console_log = log_of, heartbeat_path = beat, status = () -> :incomplete)
            t0 = time()
            result =
                supervise(spec, fast_settings(; max_retries = 3, max_stalled_retries = 2);
                    ledger_path = ledger)
            @test result.status === :stalled
            @test [a.verdict for a in result.attempts] == [:hang, :hang]
            @test all(a -> a.signal == 15 && a.stall_s >= 1.0, result.attempts)
            @test time() - t0 < 60
            recorded = TOML.parsefile(ledger)
            @test recorded["status"] == "stalled" && length(recorded["attempts"]) == 2
            @test recorded["attempts"][1]["verdict"] == "hang"
            @test isfile(log_of(1)) && isfile(log_of(2))

            # busy without progress: not a hang — left alone until busy_stall_s
            spec = WorkerSpec(;
                command = _ -> fake_worker(beat_program(beat, 1) * "; while true end"),
                console_log = log_of, heartbeat_path = beat, status = () -> :incomplete)
            result = supervise(spec, fast_settings(; max_retries = 0); ledger_path = ledger)
            @test result.status === :stalled || result.status === :retries_exhausted
            @test result.attempts[1].verdict === :livelock
            @test result.attempts[1].stall_s >= 4.0

            # ignores SIGTERM: escalation to SIGKILL (a POSIX shell with an empty
            # trap; Julia's signal thread exits on SIGTERM whatever the disposition)
            if Sys.isunix()
                immune = `sh -c "trap '' TERM; printf 'ticks = 1\\nitems_done = 0\\n' > $beat; while :; do sleep 1; done"`
                spec = WorkerSpec(; command = _ -> immune, console_log = log_of,
                    heartbeat_path = beat, status = () -> :incomplete)
                result =
                    supervise(spec, fast_settings(; max_retries = 0); ledger_path = ledger)
                @test result.attempts[1].verdict === :hang
                @test result.attempts[1].signal == 9
            end

            # never publishes a heartbeat: startup deadline
            spec = WorkerSpec(; command = _ -> fake_worker("sleep(3600)"),
                console_log = log_of, heartbeat_path = beat, status = () -> :incomplete)
            result = supervise(spec,
                fast_settings(; max_retries = 0, startup_grace_s = 1.5);
                ledger_path = ledger)
            @test result.attempts[1].verdict === :stalled_start

            # fails after finishing one item per attempt: productive attempts do
            # not count as stalled, the run completes within max_retries
            items = joinpath(dir, "items.txt")
            write(items, "")
            n_items() = countlines(items)
            program =
                "open(io -> println(io, 1), $(repr(items)), \"a\"); " *
                "exit(countlines($(repr(items))) >= 3 ? 0 : 1)"
            spec = WorkerSpec(; command = _ -> fake_worker(program), console_log = log_of,
                heartbeat_path = beat, items_done = n_items,
                status = () -> n_items() >= 3 ? :complete : :incomplete)
            result =
                supervise(spec, fast_settings(; max_retries = 3, max_stalled_retries = 1);
                    ledger_path = ledger)
            @test result.status === :complete
            @test [a.verdict for a in result.attempts] == [:failed, :failed, :complete]
            @test [a.items_after - a.items_before for a in result.attempts] == [1, 1, 1]

            # the same failure with a budget of one relaunch is exhausted
            write(items, "")
            result =
                supervise(spec, fast_settings(; max_retries = 1, max_stalled_retries = 1);
                    ledger_path = ledger)
            @test result.status === :retries_exhausted && length(result.attempts) == 2

            # input the worker cannot honour: never retried
            spec =
                WorkerSpec(; command = _ -> fake_worker("exit(2)"), console_log = log_of,
                    heartbeat_path = beat, status = () -> :incomplete)
            result = supervise(spec, fast_settings(); ledger_path = ledger)
            @test result.status === :config_error && length(result.attempts) == 1

            # deterministic failure: after max_stalled_retries attempts without a
            # new item the blocking unit is abandoned and the rest proceeds
            given_up = Ref(false)
            spec = WorkerSpec(;
                command = _ -> fake_worker(given_up[] ? "exit(0)" : "exit(3)"),
                console_log = log_of, heartbeat_path = beat,
                status = () -> given_up[] ? :partial : :incomplete,
                abandon_stalled! = () ->
                    given_up[] ? nothing : (given_up[] = true; "sweep:bad"))
            result =
                supervise(spec, fast_settings(; max_retries = 5, max_stalled_retries = 2);
                    ledger_path = ledger)
            @test result.status === :partial
            @test [a.verdict for a in result.attempts] == [:failed, :failed, :partial]
            @test result.abandoned == ["sweep:bad"]
            @test TOML.parsefile(ledger)["abandoned"] == ["sweep:bad"]
        end
    end

    @testset "continuing a run" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "config.toml")
            write(cfg_path,
                """
[pipeline]
run_2d_mapping = false
rng_seed = 5
[pipeline.sweep_settings]
n_deltas = 5
min_log_delta = 0.4
max_log_delta = 2.4
n_starts = 2
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 3.0e-3
[physics]
time_scale = 100.0
[[sweeps]]
name = "resumable"
theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
u_dir = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
[hardware]
gpu_backend = "none"
""")
            reference = run_pipeline(cfg_path, dir, "a")
            ref_sweep = joinpath(reference, "sweeps", "resumable")
            ref_lines = readlines(joinpath(ref_sweep, "checkpoint.csv"))
            @test length(ref_lines) == 2 + 5
            @test CD.Provenance.completed_stages(reference) == ["sweep:resumable"]

            # an interrupted run: the directory holds two finished items only
            partial = joinpath(dir, "b", basename(reference))
            mkpath(joinpath(partial, "sweeps", "resumable"))
            write(joinpath(partial, "sweeps", "resumable", "checkpoint.csv"),
                join(ref_lines[1:4], '\n') * "\n")
            @test run_pipeline(cfg_path, dir, "b"; run_dir = partial) == partial
            new_sweep = joinpath(partial, "sweeps", "resumable")
            @test read(joinpath(new_sweep, "results.csv")) ==
                  read(joinpath(ref_sweep, "results.csv")) # byte-identical
            new_lines = readlines(joinpath(new_sweep, "checkpoint.csv"))
            @test new_lines[1:4] == ref_lines[1:4] && length(new_lines) == 7
            @test occursin(
                "continuing: 2/5",
                read(joinpath(new_sweep, "sweep.log"), String),
            )
            @test TOML.parsefile(joinpath(partial, "metadata.toml"))["attempts"] == 1

            # continuing a complete run recomputes and overwrites nothing
            mtime_before = mtime(joinpath(new_sweep, "results.csv"))
            run_pipeline(cfg_path, dir, "b"; run_dir = partial)
            @test mtime(joinpath(new_sweep, "results.csv")) == mtime_before
            @test !isfile(joinpath(new_sweep, "results#1.csv"))
            @test TOML.parsefile(joinpath(partial, "metadata.toml"))["attempts"] == 2

            # --resume resolves the directory by configuration equality
            @test CD.Provenance.resolve_run_dir(joinpath(dir, "a"), cfg_path) ==
                  (reference, :complete)
            @test run_pipeline(cfg_path, dir, "a"; resume = true) == reference
            @test !isdir(reference * "_r2")
            # a variant that shares the run identifier is a different run
            variant = joinpath(dir, "variant.toml")
            write(variant, read(cfg_path, String) * "hessian_chunk = 3\n")
            @test run_id_from_config(variant) == run_id_from_config(cfg_path)
            fresh_dir, state = CD.Provenance.resolve_run_dir(joinpath(dir, "a"), variant)
            @test state === :fresh && fresh_dir == reference * "_r2"

            # a degenerate direction is a failed stage with a named error, and
            # concurrent launches never share a run directory
            flat = joinpath(dir, "flat.toml")
            write(flat,
                replace(read(cfg_path, String),
                    "u_dir = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]" => "u_dir = [0.0, 0.0, 0.0, 0.0, 1.0e-9, 0.0]",
                    "n_starts = 2" => "n_starts = 2\ng_uu_degenerate = 1.0e3"))
            err = try
                run_pipeline(flat, dir, "flat")
                nothing
            catch caught
                caught
            end
            @test err isa CD.PipelineStageError && err.failed_stages == ["resumable"]
            @test occursin(
                "DegenerateDirectionError",
                read(joinpath(err.run_dir, "run.log"), String),
            )
            @test isempty(CD.Provenance.completed_stages(err.run_dir))
            claimed = [
                Threads.@spawn(
                    CD.Provenance.unique_run_dir(joinpath(dir, "race"), "run_x")
                ) for _ in 1:16
            ]
            @test allunique(fetch.(claimed))
            @test CD.Provenance.note_figure_failure!(reference, "sweep:resumable") ==
                  ["sweep:resumable"]
            @test TOML.parsefile(joinpath(reference, "metadata.toml"))["figure_failures"] ==
                  ["sweep:resumable"]

            # the supervisor drives the real worker; a second call is a no-op
            worker = joinpath(dirname(@__DIR__), "scripts", "run_pipeline.jl")
            write(variant,
                read(cfg_path, String) *
                "\n[supervision]\npoll_interval_s = 0.5\nidle_stall_s = 120.0\n")
            run_dir, result = supervise_pipeline(variant, dir, "c"; worker_script = worker,
                julia = JULIA_BIN, threads = "2", worker_env = CLEAN_ENV)
            @test result.status === :complete && length(result.attempts) == 1
            @test read(joinpath(run_dir, "sweeps", "resumable", "results.csv")) ==
                  read(joinpath(ref_sweep, "results.csv"))
            @test isfile(joinpath(run_dir, "supervision.toml"))
            @test isfile(joinpath(run_dir, "logs", "attempt_1.console.log"))
            @test CD.Heartbeat.read_heartbeat(joinpath(run_dir, "heartbeat.toml")).ticks > 0
            _, again = supervise_pipeline(variant, dir, "c"; worker_script = worker,
                julia = JULIA_BIN, threads = "2", worker_env = CLEAN_ENV)
            @test again.status === :complete && isempty(again.attempts)
        end
    end
end
