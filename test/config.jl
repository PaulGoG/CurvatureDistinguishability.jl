# Configuration validation and run provenance. Included by runtests.jl.

@testset "Config validation guardrails" begin
    function write_cfg(dir, extra)
        path = joinpath(dir, "config.toml")
        base = """
        [grid]
        T_obs = 1.0e5
        f_min = 1.0e-3
        f_max = 2.0e-3
        """
        write(path, base * extra)
        return path
    end
    mktempdir() do dir
        cfg = load_and_validate_config(write_cfg(dir, ""))
        @test cfg isa PipelineSettings
        @test cfg.optimizer === :ipnewton
        # T_obs = 1e5 s → nearest Robson column is 6 months
        @test cfg.noise.confusion_knee_freq == 0.00258

        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[pipeline]\noptimizer = \"sgd\"\n"))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[pipeline.sweep_settings]\nn_deltas = 1\n"))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[parameter_bounds]\nspin1 = [2.0, 1.0]\n"))
        # require_gpu contradicts a forced CPU backend; with a GPU request
        # it is a plain flag
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[hardware]\ngpu_backend = \"none\"\nrequire_gpu = true\n"))
        @test load_and_validate_config(
            write_cfg(dir, "[hardware]\ngpu_backend = \"auto\"\nrequire_gpu = true\n"),
        ).require_gpu
        @test !load_and_validate_config(write_cfg(dir, "")).require_gpu
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(
                dir,
                "[[sweeps]]\nname = \"bad/name\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\n",
            ))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(
                dir,
                "[[maps]]\nname = \"m\"\nparam_x = 2\nparam_y = 2\ntheta_0 = [1,1,1,0,0,0]\n",
            ))
        # case names that would escape the case directory are rejected
        for bad in (".", "..", "...")
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[maps]]\nname = \"$bad\"\nparam_x = 2\nparam_y = 3\ntheta_0 = [1,1,1,0,0,0]\n",
                ))
        end
        @test CD.Config.fs_safe("a.b-c_1") && !CD.Config.fs_safe("a/b")
        # remaining validation branches: symmetric mass ratio, grid size,
        # progress-log cadence
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[physics]\neta = 0.3\n"))
        @test load_and_validate_config(write_cfg(dir, "[physics]\neta = 0.2\n")).wp.eta ==
              0.2
        # post-Newtonian order of the phase: 1.5 (default) to 4.5 in half
        # orders, as a number, integer or float
        @test load_and_validate_config(write_cfg(dir, "")).wp.phase_pn_order == 1.5
        for (order, value) in (("2", 2.0), ("2.0", 2.0), ("2.5", 2.5), ("3", 3.0),
            ("3.5", 3.5), ("4", 4.0), ("4.5", 4.5))
            @test load_and_validate_config(
                write_cfg(dir, "[physics]\nphase_pn_order = $order\n"),
            ).wp.phase_pn_order == value
        end
        for bad in ("2.25", "1", "5.0", "\"2PN\"")
            @test_throws "[physics].phase_pn_order" load_and_validate_config(
                write_cfg(dir, "[physics]\nphase_pn_order = $bad\n"))
        end
        @test load_and_validate_config(write_cfg(dir, "")).fit_coordinates === :parameters
        @test load_and_validate_config(
            write_cfg(dir, "[pipeline]\nfit_coordinates = \"whitened\"\n"),
        ).fit_coordinates === :whitened
        @test_throws "[pipeline].fit_coordinates" load_and_validate_config(
            write_cfg(dir, "[pipeline]\nfit_coordinates = \"fisher\"\n"))
        @test_throws "requires optimizer" load_and_validate_config(
            write_cfg(dir,
                "[pipeline]\nfit_coordinates = \"whitened\"\noptimizer = \"lbfgs_box\"\n"))
        # a value of the wrong type is a configuration error naming the
        # key, not a MethodError: strings, sections and arrays of tables
        for (bad, key) in (
            ("[pipeline]\noptimizer = 3\n", "optimizer"),
            ("[hardware]\ngpu_backend = 1\n", "gpu_backend"),
            ("[[sweeps]]\nname = 7\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\n",
                "name"),
            ("[sweeps]\nname = \"s\"\n", "sweeps"),
            ("[[noise]]\nconfusion_enabled = true\n", "noise"),
        )
            @test_throws ArgumentError load_and_validate_config(write_cfg(dir, bad))
            @test_throws key load_and_validate_config(write_cfg(dir, bad))
        end
        # [supervision]: absent keys take the defaults of the settings
        # constructor, present ones override them one by one
        sup_default = load_and_validate_config(write_cfg(dir, "")).supervision
        @test sup_default == CD.Supervision.SupervisionSettings()
        sup = load_and_validate_config(
            write_cfg(dir, "[supervision]\nmax_retries = 7\nbackoff_s = 2.5\n"),
        ).supervision
        @test sup.max_retries == 7 && sup.backoff_s == 2.5
        @test sup.poll_interval_s == sup_default.poll_interval_s
        @test_throws "[supervision].max_retries" load_and_validate_config(
            write_cfg(dir, "[supervision]\nmax_retries = -1\n"))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[supervision]\nmax_retries = 1.5\n"))
        # a Boolean is not a number and an infinite or undefined value is
        # not a setting: both are refused with the key named
        for (bad, key) in (
            ("[supervision]\nmax_retries = true\n", "[supervision].max_retries"),
            ("[supervision]\nbackoff_s = inf\n", "[supervision].backoff_s"),
            ("[physics]\nmass_scale = inf\n", "[physics].mass_scale"),
            ("[physics]\neta = nan\n", "[physics].eta"),
            ("[mapping]\nn_angles = true\n", "[mapping].n_angles"),
            ("[physics]\nphase_pn_order = true\n", "[physics].phase_pn_order"),
        )
            @test_throws key load_and_validate_config(write_cfg(dir, bad))
        end
        # an explicit run directory that holds the snapshot of another
        # configuration is refused before anything is computed
        other = joinpath(dir, "other_run")
        mkpath(other)
        cp(write_cfg(dir, "[physics]\neta = 0.2\n"), joinpath(other, "config.toml"))
        @test_throws "another configuration" run_pipeline(
            write_cfg(dir, "[physics]\neta = 0.21\n"), dir, "out"; run_dir = other)
        @test readdir(other) == ["config.toml"]
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[physics]\necliptic_latitude = 2.0\n"))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[physics]\ncutoff_width = 0.0\n"))
        cfg_arm =
            load_and_validate_config(write_cfg(dir, "[noise]\narm_length = 5.0e9\n"))
        @test cfg_arm.wp.arm_length == 5.0e9 # one arm length for noise and response
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[monitoring]\nprogress_log_fraction = 2.0\n"))
        tiny = joinpath(dir, "tiny.toml")
        write(tiny, "[grid]\nT_obs = 100.0\nf_min = 1.0e-3\nf_max = 2.0e-3\n")
        @test_throws ArgumentError load_and_validate_config(tiny)
        # base point outside physical bounds
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(
                dir,
                "[[maps]]\nname = \"m\"\nparam_x = 5\nparam_y = 6\ntheta_0 = [1,1,1,0,1.5,0]\n",
            ))
        # unknown keys abort and the message names the key (typo protection);
        # odd n_angles warns and rounds
        @test_throws "Unknown configuration key 'n_anglse'" load_and_validate_config(
            write_cfg(dir, "[mapping]\nn_anglse = 100\n"))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[hardware]\nrequire_gp = true\n"))
        cfg_odd =
            @test_logs (:warn, r"odd; rounding up") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[mapping]\nn_angles = 33\n"))
        @test cfg_odd.map_n_angles == 34
        @test cfg_odd.corner_bisect_iters == 25 # safe default: corners exact
        cfg_nb = load_and_validate_config(
            write_cfg(dir, "[mapping]\ncorner_bisect_iters = 0\n"))
        @test cfg_nb.corner_bisect_iters == 0 # explicit opt-out allowed
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[mapping]\ncorner_bisect_iters = -1\n"))

        # amp_ratio and multi-start guardrails; an amplitude component in
        # the sweep direction is rejected (amp_ratio carries it)
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(
                dir,
                "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0.2,1,0,0,0,0]\n",
            ))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(
                dir,
                "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\namp_ratio = -1.0\n",
            ))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(
                dir,
                "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0.5,1,0,0,0,0]\namp_ratio = 0.5\n",
            ))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[pipeline.sweep_settings]\nn_starts = 0\n"))
        cfg_ms = load_and_validate_config(
            write_cfg(
                dir,
                "[pipeline]\nrng_seed = 7\n[pipeline.sweep_settings]\nn_starts = 3\ng_tol = 1e-9\nmax_iterations = 300\n[hardware]\nhessian_chunk = 3\n",
            ))
        @test cfg_ms.n_starts == 3 && cfg_ms.rng_seed == 7
        @test cfg_ms.g_tol == 1e-9 && cfg_ms.max_iterations == 300
        @test cfg_ms.hessian_chunk == 3
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[hardware]\nhessian_chunk = 9\n"))
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[pipeline.sweep_settings]\ng_tol = 0.0\n"))

        # safe-by-default optimizer tolerances; monitoring opt-in (off)
        cfg_def = load_and_validate_config(write_cfg(dir, ""))
        @test cfg_def.g_tol == 1e-10 && cfg_def.max_iterations == 100
        @test cfg_def.monitoring_enabled == false
        cfg_mon =
            load_and_validate_config(write_cfg(dir, "[monitoring]\nenabled = true\n"))
        @test cfg_mon.monitoring_enabled == true
        @test_logs (:warn, r"tighter than the numerical precision floor") match_mode =
            :any load_and_validate_config(
            write_cfg(dir, "[pipeline.sweep_settings]\ng_tol = 1e-12\n"))
        @test_logs (:warn, r"floor fits exhaust the full cap") match_mode = :any load_and_validate_config(
            write_cfg(dir, "[pipeline.sweep_settings]\nmax_iterations = 1000\n"))

        cfg_vs =
            load_and_validate_config(write_cfg(dir, "[safety]\nmax_vram_gb = 6.0\n"))
        @test cfg_vs.max_vram_gb == 6.0

        # all shipped configurations must validate as-is (production
        # variants carry the full 7-sweep campaign; quickstart carries 2)
        for shipped in filter(f -> endswith(f, ".toml"),
            readdir(joinpath(dirname(@__DIR__), "configs")))
            cfg_ship = load_and_validate_config(
                joinpath(dirname(@__DIR__), "configs", shipped))
            if startswith(shipped, "production")
                @test cfg_ship.n_deltas == 30 && length(cfg_ship.sweeps) == 7
            else
                @test length(cfg_ship.sweeps) == 2
            end
        end
        # analysis tunables: defaults, override roundtrip, validation
        @test cfg_def.floor_detection_ratio == 2.0
        @test cfg_def.min_log_delta == -3.0 && cfg_def.max_log_delta == 2.5
        @test cfg_def.correction_fit_max_departure == 0.3
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir,
                "[pipeline.sweep_settings]\ncorrection_fit_max_departure = 0.0\n"),
        )
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir,
                "[pipeline.sweep_settings]\nmin_log_delta = 0.5\nmax_log_delta = 0.3\n",
            ),
        )
        @test cfg_def.secondary_minimum_gain == 1.5
        @test cfg_def.unbounded_cap_factor == 5.0
        @test cfg_def.residual_spectrum_windows == 600
        cfg_tun = load_and_validate_config(
            write_cfg(dir,
                "[pipeline.sweep_settings]\nfloor_detection_ratio = 3.0\n" *
                "[mapping]\nunbounded_cap_factor = 8.0\n"),
        )
        @test cfg_tun.floor_detection_ratio == 3.0
        @test cfg_tun.unbounded_cap_factor == 8.0
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[pipeline.sweep_settings]\nfloor_detection_ratio = 1.0\n"))

        cfg_quick = load_and_validate_config(
            joinpath(dirname(@__DIR__), "configs", "quickstart.toml"))
        @test length(cfg_quick.sweeps) == 2 && length(cfg_quick.maps) == 2
        @test cfg_quick.monitoring_enabled
        # work items parse once into typed specs with resolved defaults
        @test cfg_quick.sweeps isa Vector{SweepSpec}
        @test cfg_quick.maps isa Vector{MapSpec}
        @test cfg_quick.sweeps[1].rho_thresh == 1.0 # pipeline-wide default
        @test cfg_quick.sweeps[1].amp_ratio == 1.0
        @test cfg_quick.maps[1].n_angles == 256 # [mapping].n_angles fallback
        cfg_na =
            @test_logs (:warn, r"odd; rounding up") match_mode = :any load_and_validate_config(
                write_cfg(
                    dir,
                    "[[maps]]\nname = \"m\"\nparam_x = 5\nparam_y = 6\n" *
                    "theta_0 = [1,1,1,0,0,0]\nn_angles = 33\n",
                ))
        @test cfg_na.maps[1].n_angles == 34 # per-map override resolved even

        # instrumental-noise overrides thread through to NoiseParams and
        # must be strictly positive
        cfg_n = load_and_validate_config(
            write_cfg(dir, "[noise]\narm_length = 5.0e9\nacc_amplitude = 2.4e-15\n"))
        @test cfg_n.noise.arm_length == 5.0e9
        @test cfg_n.noise.acc_amplitude == 2.4e-15
        @test cfg_n.noise.oms_amplitude == 1.5e-11
        @test_throws ArgumentError load_and_validate_config(
            write_cfg(dir, "[noise]\narm_length = 0.0\n"))
    end
end

@testset "Provenance: backup-before-overwrite" begin
    mktempdir() do dir
        target = joinpath(dir, "results.csv")
        write(target, "first")
        # each call moves the existing file to the next free #k backup
        @test backup_existing!(target) == target
        write(target, "second")
        @test backup_existing!(target) == target
        write(target, "third")
        @test read(joinpath(dir, "results#1.csv"), String) == "first"
        @test read(joinpath(dir, "results#2.csv"), String) == "second"
        @test read(target, String) == "third"
        # a fresh path is returned untouched, creating nothing
        @test backup_existing!(joinpath(dir, "new.csv")) == joinpath(dir, "new.csv")
        @test !isfile(joinpath(dir, "new.csv"))
        # rerun directories are suffixed, never reused
        first_dir = CD.Provenance.unique_run_dir(dir, "run_abc")
        second_dir = CD.Provenance.unique_run_dir(dir, "run_abc")
        third_dir = CD.Provenance.unique_run_dir(dir, "run_abc")
        @test basename(first_dir) == "run_abc"
        @test basename(second_dir) == "run_abc_r2" && isdir(second_dir)
        @test basename(third_dir) == "run_abc_r3"
    end
    # one process per run directory: a held lock refuses a second holder with the
    # directory named, and is free again once closed
    mktempdir() do dir
        lock = CD.Provenance.acquire_run_lock(dir)
        @test isfile(joinpath(dir, "run.pid"))
        @test_throws "in use by another process" CD.Provenance.acquire_run_lock(dir)
        close(lock)
        @test !isfile(joinpath(dir, "run.pid"))
        close(CD.Provenance.acquire_run_lock(dir))
        # the lock file of a holder that no longer runs on this host (a worker
        # killed by the supervisor) is reclaimed; one of another host is not
        path = joinpath(dir, "run.pid")
        dead = "2147483646 $(gethostname())"
        write(path, dead)
        close(CD.Provenance.acquire_run_lock(dir))
        @test !isfile(path * ".reclaim")
        write(path, "2147483646 elsewhere-host")
        @test_throws "another host cannot be checked" CD.Provenance.acquire_run_lock(dir)
        rm(path)
        # a reclaim in progress elsewhere (its guard held) is not raced
        write(path, dead)
        guard = CD.Provenance.acquire_run_lock(dir; name = "run.pid.reclaim")
        @test_throws "in use by another process" CD.Provenance.acquire_run_lock(dir)
        close(guard)
        @test !isfile(path * ".reclaim")
        close(CD.Provenance.acquire_run_lock(dir))
        # a process that read the stale record late, after another reclaimed the
        # file, leaves the live lock alone
        live = CD.Provenance.acquire_run_lock(dir)
        @test CD.Provenance.reclaim_stale_lock(path, dead) === nothing
        @test CD.Provenance.lock_holder(path) == "$(getpid()) $(gethostname())"
        close(live)
        # a guard left by a reclaimer that was killed is named in the error
        write(path, dead)
        write(path * ".reclaim", dead)
        msg = try
            CD.Provenance.acquire_run_lock(dir)
            ""
        catch err
            err isa ArgumentError || rethrow()
            err.msg
        end
        @test occursin(path * " ", msg) && occursin(path * ".reclaim ", msg)
        rm(path * ".reclaim")
        close(CD.Provenance.acquire_run_lock(dir))
        @test CD.Provenance.lock_holder(path) == "unknown"
        # supervisor and worker locks are independent
        s = CD.Provenance.acquire_run_lock(dir; name = "supervisor.pid")
        w = CD.Provenance.acquire_run_lock(dir)
        close(w)
        close(s)
    end
    # per-attempt provenance: each attempt is recorded, a change of host, Julia
    # version or manifest against attempt 1 is reported
    mktempdir() do dir
        CD.Provenance.record_attempt!(dir, 1; git = "abc", backend = "CPU")
        meta = TOML.parsefile(joinpath(dir, "metadata.toml"))
        rec = meta["attempt_records"]["1"]
        @test rec["git"] == "abc" && rec["backend"] == "CPU"
        @test rec["julia_version"] == string(VERSION) && rec["hostname"] == gethostname()
        @test haskey(rec, "manifest_sha256") && haskey(rec, "started")
        CD.Provenance.record_attempt!(dir, 2; git = "def", backend = "CPU")
        meta = TOML.parsefile(joinpath(dir, "metadata.toml"))
        @test sort(collect(keys(meta["attempt_records"]))) == ["1", "2"]
        # a different host in attempt 1 is reported for attempt 3
        meta["attempt_records"]["1"]["hostname"] = "elsewhere"
        open(io -> TOML.print(io, meta), joinpath(dir, "metadata.toml"), "w")
        @test_logs (:warn, r"hostname") match_mode = :any CD.Provenance.record_attempt!(dir,
            3; git = "def", backend = "CPU")
    end
end

@testset "Provenance: comment-invariant run identifiers" begin
    mktempdir() do dir
        a = joinpath(dir, "a.toml")
        b = joinpath(dir, "b.toml")
        c = joinpath(dir, "c.toml")
        write(a, "[grid]\nT_obs = 1.0e5\nf_min = 1.0e-3\n")
        # same values: reordered keys, comments, blank lines
        write(
            b,
            "# leading comment\n[grid]\n\nf_min = 1.0e-3 # trailing\nT_obs = 1.0e5\n",
        )
        # one value changed
        write(c, "[grid]\nT_obs = 2.0e5\nf_min = 1.0e-3\n")
        @test run_id_from_config(a) == run_id_from_config(b)
        @test run_id_from_config(a) != run_id_from_config(c)
        @test startswith(run_id_from_config(a), "run_")
    end
end

@testset "Provenance: base + overlay configurations" begin
    mktempdir() do dir
        base = joinpath(dir, "base.toml")
        overlay = joinpath(dir, "overlay.toml")
        mono = joinpath(dir, "monolithic.toml")
        write(
            base,
            """
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 2.0e-3
[hardware]
gpu_backend = "none"
max_threads = 4
hessian_chunk = 0
[[sweeps]]
name = "a"
theta_0 = [1, 1, 1, 0, 0, 0]
u_dir = [0, 1, 0, 0, 0, 0]
""",
        )
        write(
            overlay,
            """
base_config = "base.toml"
[hardware]
gpu_backend = "auto"
hessian_chunk = 3
[[sweeps]]
name = "b"
theta_0 = [1, 1, 1, 0, 0, 0]
u_dir = [0, 0, 1, 0, 0, 0]
""",
        )
        write(
            mono,
            """
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 2.0e-3
[hardware]
gpu_backend = "auto"
max_threads = 4
hessian_chunk = 3
[[sweeps]]
name = "b"
theta_0 = [1, 1, 1, 0, 0, 0]
u_dir = [0, 0, 1, 0, 0, 0]
""",
        )
        # merge semantics: sub-tables recurse, scalars and arrays of
        # tables replace, the base_config key is stripped
        eff = effective_config(overlay)
        @test !haskey(eff, "base_config")
        @test eff["hardware"]["max_threads"] == 4
        @test eff["hardware"]["gpu_backend"] == "auto"
        @test length(eff["sweeps"]) == 1 && eff["sweeps"][1]["name"] == "b"
        @test eff["grid"]["T_obs"] == 1.0e5
        # identity: an overlay hashes exactly like the equivalent monolithic file
        @test run_id_from_config(overlay) == run_id_from_config(mono)
        @test run_id_from_config(overlay) != run_id_from_config(base) # sweeps differ
        # execution sections never enter the identity: the same physical
        # case on another machine (threads, backend, budgets, monitoring)
        # keeps its run identifier
        exec_variant = joinpath(dir, "exec_variant.toml")
        write(exec_variant, read(mono, String) * """
[safety]
max_ram_gb = 200.0
max_vram_gb = 48.0
[monitoring]
enabled = true
""")
        @test run_id_from_config(exec_variant) == run_id_from_config(mono)
        hw_variant = joinpath(dir, "hw_variant.toml")
        write(
            hw_variant,
            replace(read(mono, String), "max_threads = 4" => "max_threads = 64",
                "gpu_backend = \"auto\"" => "gpu_backend = \"none\""),
        )
        @test run_id_from_config(hw_variant) == run_id_from_config(mono)
        @test !any(
            haskey(identity_config(effective_config(exec_variant)), s)
            for s in ("hardware", "safety", "monitoring")
        )
        @test haskey(identity_config(effective_config(exec_variant)), "grid")
        cfg = load_and_validate_config(overlay)
        @test cfg.gpu_backend === :auto && cfg.max_threads == 4 &&
              cfg.hessian_chunk == 3
        @test [s.name for s in cfg.sweeps] == ["b"]
        # the snapshot of an overlay is the self-contained merged table and
        # rehashes to the run's identifier; plain files are copied verbatim
        run_dir = mktempdir(dir)
        snap = CD.Provenance.snapshot_config(overlay, run_dir)
        @test !haskey(TOML.parsefile(snap), "base_config")
        @test run_id_from_config(snap) == run_id_from_config(overlay)
        @test_throws ArgumentError CD.Provenance.snapshot_config(overlay, run_dir)
        run_dir2 = mktempdir(dir)
        snap2 = CD.Provenance.snapshot_config(mono, run_dir2)
        @test read(snap2, String) == read(mono, String)
        # one overlay level only; a missing base fails loudly
        nested = joinpath(dir, "nested.toml")
        write(nested, "base_config = \"overlay.toml\"\n")
        @test_throws ArgumentError effective_config(nested)
        @test_throws ArgumentError load_and_validate_config(nested)
        absent = joinpath(dir, "absent.toml")
        write(absent, "base_config = \"no_such_file.toml\"\n")
        @test_throws ArgumentError effective_config(absent)
        write(absent, "base_config = 3\n")
        @test_throws ArgumentError effective_config(absent)
    end
    # the shipped GPU variants are thin execution-only overlays on their
    # bases ([hardware], and [safety] for the device budget)
    configs = joinpath(dirname(@__DIR__), "configs")
    for (overlay, base) in (("production_gpu.toml", "production_cpu.toml"),
        ("quickstart_gpu.toml", "quickstart.toml"))
        raw = TOML.parsefile(joinpath(configs, overlay))
        @test raw["base_config"] == base
        @test collect(keys(raw)) ⊆ ["base_config", "hardware", "safety"]
        @test raw["hardware"]["require_gpu"] === true
        eff = effective_config(joinpath(configs, overlay))
        @test eff["sweeps"] == TOML.parsefile(joinpath(configs, base))["sweeps"]
        # same physics, other execution settings: same run identifier
        @test run_id_from_config(joinpath(configs, overlay)) ==
              run_id_from_config(joinpath(configs, base))
    end
    # phase-order overlays: one quickstart and one production-map overlay per
    # order, each at the order its name gives
    pn_dir = joinpath(configs, "pn_orders")
    for order in CD.Physics.PHASE_PN_ORDERS
        tag = replace(replace(string(order), r"\.0$" => ""), "." => "p")
        quick = load_and_validate_config(joinpath(pn_dir, "quickstart_pn$tag.toml"))
        @test quick.wp.phase_pn_order == order && quick.fit_coordinates === :whitened
        @test length(quick.sweeps) == 2
        maps = load_and_validate_config(joinpath(pn_dir, "production_maps_pn$tag.toml"))
        @test maps.wp.phase_pn_order == order && !maps.run_1d_sweeps
        @test length(maps.maps) == 4 && all(m -> m.n_angles == 1024, maps.maps)
    end
    @test length(filter(f -> endswith(f, ".toml"), readdir(pn_dir))) ==
          2 * length(CD.Physics.PHASE_PN_ORDERS)
    # campaign configurations: every shipped file validates, the Hessian
    # chunk never enters the run identity, and each single-sweep overlay
    # reproduces exactly one case of the production base
    campaigns = joinpath(configs, "campaigns")
    campaign_files = filter(f -> endswith(f, ".toml"), readdir(campaigns))
    required = vcat(
        ["benchmark_base.toml", "windowed_low_mass_maps.toml", "cpu_full.toml",
            "gpu_stress_benchmark.toml", "gpu_stress_chunk0.toml",
            "gpu_stress_chunk1.toml", "gpu_stress_chunk2.toml",
            "gpu_sweeps_chunk0.toml", "gpu_sweeps_chunk1.toml",
            "gpu_sweeps_chunk2.toml", "gpu_sweeps_chunk3.toml"],
        ["sweep$(n)_chunk$(c).toml" for n in 1:7 for c in 0:3])
    @test required ⊆ campaign_files
    for f in campaign_files
        @test load_and_validate_config(joinpath(campaigns, f)) isa PipelineSettings
    end
    # the campaign is frozen on its own base: it keeps the identifiers it ran
    # under, whatever the production configuration becomes
    base = load_and_validate_config(joinpath(campaigns, "benchmark_base.toml"))
    @test base.wp.observation_time == 0
    @test load_and_validate_config(joinpath(configs, "production_cpu.toml")).wp.observation_time ==
          base.T_obs
    for (file, id) in ("cpu_full.toml" => "run_5c0a7409",
        "gpu_stress_benchmark.toml" => "run_9f93c3af",
        "gpu_sweeps_chunk2.toml" => "run_96aad190",
        "windowed_low_mass_maps.toml" => "run_b2538c19")
        @test run_id_from_config(joinpath(campaigns, file)) == id
    end
    same_sweep(a, b) =
        all(getfield(a, f) == getfield(b, f) for f in fieldnames(SweepSpec))
    sweep_ids = String[]
    for n in 1:7
        ids = [
            run_id_from_config(joinpath(campaigns, "sweep$(n)_chunk$(c).toml"))
            for c in 0:3
        ]
        @test length(unique(ids)) == 1
        push!(sweep_ids, first(ids))
        for c in 0:3
            single =
                load_and_validate_config(
                    joinpath(campaigns, "sweep$(n)_chunk$(c).toml"),
                )
            @test length(single.sweeps) == 1 &&
                  same_sweep(single.sweeps[1], base.sweeps[n])
            @test single.hessian_chunk == c && single.require_gpu
        end
    end
    @test sweep_ids == ["run_ca598507", "run_5d0a1f8a", "run_253f6ad6", "run_a787b9b3",
        "run_9c4f5e27", "run_cb2b9055", "run_fbc895db"]
    @test length(
        unique(
            run_id_from_config(joinpath(campaigns, f)) for f in
            ("gpu_stress_benchmark.toml", "gpu_stress_chunk0.toml",
                "gpu_stress_chunk1.toml", "gpu_stress_chunk2.toml")
        ),
    ) == 1
    @test length(
        unique(
            run_id_from_config(joinpath(campaigns,
                "gpu_sweeps_chunk$(c).toml")) for c in 0:3
        ),
    ) == 1
end
