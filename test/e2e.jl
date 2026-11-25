# End-to-end runs of the pipeline and its scripts. Included by runtests.jl.

@testset "End-to-end minimal pipeline" begin
    mktempdir() do dir
        cfg_path = joinpath(dir, "config.toml")
        write(
            cfg_path,
            """
[pipeline]
optimizer = "ipnewton"
rng_seed = 11
[pipeline.sweep_settings]
n_deltas = 6
min_log_delta = 0.4
max_log_delta = 2.7
n_starts = 2
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 3.0e-3
[physics]
time_scale = 100.0
ecliptic_longitude = 3.1415
ecliptic_latitude = 0.5238
polarization = 0.785
[mapping]
n_angles = 32
max_refine_levels = 3
[safety]
max_ram_gb = 8.0
[[sweeps]]
name = "mini_sweep"
theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
u_dir = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
[[sweeps]]
name = "mini_unequal"
theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
u_dir = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
amp_ratio = 0.5
[[maps]]
name = "mini_spin_map"
param_x = 5
param_y = 6
rho_thresh = 1.0
theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
[[maps]]
name = "mini_mass_time"
param_x = 2
param_y = 3
theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
""",
        )
        out_base = run_pipeline(cfg_path, dir, "outputs")
        @test isdir(out_base)
        @test isfile(joinpath(out_base, "config.toml"))
        @test isfile(joinpath(out_base, "run.log"))
        meta = TOML.parsefile(joinpath(out_base, "metadata.toml"))
        # hardware-provenance sidecar: host fingerprint always present
        @test isfile(joinpath(out_base, "hardware.txt"))
        @test occursin("BLAS threads", read(joinpath(out_base, "hardware.txt"), String))
        @test haskey(meta, "git") && haskey(meta, "finished")
        @test meta["failed_stages"] == ""
        # the resolved manifest travels with the run and is never overwritten
        @test isfile(joinpath(out_base, "Manifest.toml"))
        @test_throws ArgumentError snapshot_manifest(out_base)

        sdir = joinpath(out_base, "sweeps", "mini_sweep")
        res = CSV.read(joinpath(sdir, "results.csv"), DataFrame)
        @test nrow(res) == 6
        @test all(
            hasproperty.(Ref(res), [:BestFit_Spin1, :Converged, :Iterations, :AtBound]),
        )
        @test all(res.D2_Numerical .>= 0)
        # multi-start bookkeeping (n_starts = 2): best-of never worse than canonical
        @test all(res.Starts .== 2)
        @test all(res.MultiStartGain .>= 1.0 - 1e-12)
        for f in ("scaling_plot.pdf", "scaling_plot.png", "residual_plot.pdf",
            "residual_plot.png", "residual_spectrum.csv", "sweep_meta.toml")
            @test isfile(joinpath(sdir, f))
        end
        meta_s = TOML.parsefile(joinpath(sdir, "sweep_meta.toml"))
        @test haskey(meta_s, "c1") && haskey(meta_s, "delta_valid")

        # unequal-amplitude sweep: in the perturbative (small-δ) regime the
        # A_harm prefactor (2q/(1+q))² must make the ratio ≈ 1; a missing
        # prefactor would show as ratio ≈ 2.25. (Only the small-δ rows: for
        # q ≠ 1 the midpoint symmetry that suppresses the odd O(δ⁵) term is
        # absent, so the higher-order departure sets in much earlier.)
        res_u = CSV.read(
            joinpath(out_base, "sweeps", "mini_unequal", "results.csv"),
            DataFrame,
        )
        ratio_u = res_u.D2_Numerical[1:3] ./ res_u.D2_Theoretical[1:3]
        @test all(0.85 .< ratio_u .< 1.15)
        meta_u = TOML.parsefile(
            joinpath(out_base, "sweeps", "mini_unequal", "sweep_meta.toml"),
        )
        @test meta_u["amp_ratio"] == 0.5
        # δ_min inverts the amplitude-prefactored law (p/16) K δ⁴ = ρ²
        @test meta_u["delta_min"] ≈
              (16 * meta_u["rho_sq"] / ((2 * 0.5 / 1.5)^2 * meta_u["K_u_norm"]))^(1 / 4) rtol =
            1e-12

        for map_name in ("mini_spin_map", "mini_mass_time")
            mdir = joinpath(out_base, "maps", map_name)
            cm = CSV.read(joinpath(mdir, "confusion_contour.csv"), DataFrame)
            n = nrow(cm)
            @test iseven(n)
            half = n ÷ 2
            # the mathematical radius is exactly centrally symmetric by
            # construction (mirrored evaluation); the capped radius need
            # not be, since the prior box may be asymmetric about the base point
            @test cm.R_Math[1:half] == cm.R_Math[(half+1):end]
            @test cm.K_Raw[1:half] == cm.K_Raw[(half+1):end]
            @test all(cm.Angle[(half+1):end] .≈ cm.Angle[1:half] .+ π)
            @test isfile(joinpath(mdir, "confusion_zone.pdf"))
            @test isfile(joinpath(mdir, "confusion_zone.png"))
            # every vertex sitting on a prior wall carries the flag (the
            # bisected crossover vertices included), so the drawn wall
            # segment reaches its corners
            on_wall = isfinite.(cm.R_Box) .& (cm.R_Capped .== cm.R_Box)
            @test all(cm.Prior_Limited[on_wall])
            @test !any(cm.Prior_Limited[.!on_wall])
            # exact prior-crossover corners: at every capped/uncapped
            # transition along the boundary polygon, one of the two rows
            # must be the bisection-inserted vertex with r_math ≈ r_box;
            # otherwise the polygon chord chamfers the zone corner
            n_trans = 0
            for i in 1:n
                j = mod1(i + 1, n)
                cm.Prior_Limited[i] == cm.Prior_Limited[j] && continue
                n_trans += 1
                rel = [
                    abs(cm.R_Math[k] - cm.R_Box[k]) / cm.R_Box[k]
                    for
                    k in (i, j) if isfinite(cm.R_Box[k]) && isfinite(cm.R_Math[k])
                ]
                @test any(<(1e-3), rel)
            end
            count(cm.Prior_Limited) in (0, n) || @test n_trans >= 2
        end

        # exact box-corner vertex: whenever both walls of the mass-time box
        # are active, the boundary must pass through their corner exactly
        cmt = CSV.read(
            joinpath(out_base, "maps", "mini_mass_time", "confusion_contour.csv"),
            DataFrame)
        x_wall = abs.(cmt.X_Bound .+ 1.5) .< 1e-9
        y_wall = abs.(cmt.Y_Bound .+ 2.0) .< 1e-9
        if any(x_wall) && any(y_wall)
            @test any(x_wall .& y_wall)
        end

        # the spin map must be capped inside the physical box
        cm = CSV.read(
            joinpath(out_base, "maps", "mini_spin_map", "confusion_contour.csv"),
            DataFrame)
        @test all(-1.8 - 1e-9 .<= cm.X_Bound .<= 0.2 + 1e-9)
        @test all(-1.8 - 1e-9 .<= cm.Y_Bound .<= 0.2 + 1e-9)
        @test count(cm.Prior_Limited) > 0 # χ_a degeneracy → prior-limited directions
        @test all(cm.R_Capped .<= cm.R_Box .+ 1e-12)
        @test hasproperty(cm, :Dir_Cos) && hasproperty(cm, :Dir_Sin)

        # threshold-rescaled replot from persisted K (no recomputation).
        # Plain binary (not julia_cmd(): --check-bounds would recompile the
        # plotting stack) and a scrubbed environment (Pkg.test exports its
        # sandbox via JULIA_LOAD_PATH/JULIA_PROJECT, which the child would
        # otherwise resolve against).
        replot_script = joinpath(dirname(@__DIR__), "scripts", "replot.jl")
        jlbin = Base.julia_cmd().exec[1]
        cmd = addenv(`$jlbin --startup-file=no $replot_script $out_base --rho 2.0`,
            "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
        run(pipeline(cmd, stdout = devnull, stderr = devnull))
        rcsv =
            joinpath(out_base, "maps", "mini_spin_map", "confusion_contour_rho2p0.csv")
        @test isfile(rcsv)
        cm2 = CSV.read(rcsv, DataFrame)
        @test all(cm2.R_Capped .<= cm2.R_Box .+ 1e-12)
        # where neither threshold is box-capped, radii scale exactly by √2
        free = .!cm.Prior_Limited .& .!cm2.Prior_Limited
        if any(free)
            @test all(
                isapprox.(
                    cm2.R_Capped[free],
                    sqrt(2.0) .* cm.R_Capped[free];
                    rtol = 1e-10,
                ),
            )
        end
        @test isfile(
            joinpath(out_base, "maps", "mini_spin_map", "confusion_zone_rho2p0.png"),
        )

        # RunFigures: figures rebuild from the persisted CSVs alone, and the
        # library ρ-rescale reproduces the replot child's contour (below)
        cases = run_cases(out_base)
        @test cases.sweeps == ["mini_sweep", "mini_unequal"]
        @test cases.maps == ["mini_mass_time", "mini_spin_map"]
        figs = sweep_figures(out_base, "mini_sweep")
        @test figs.residual !== nothing && figs.suffix == ""
        # rescaling to the run's own threshold reproduces the persisted δ_min,
        # amplitude prefactor included (mini_unequal has A₂/A₁ = 0.5)
        for case in cases.sweeps
            sm = TOML.parsefile(joinpath(out_base, "sweeps", case, "sweep_meta.toml"))
            rescaled = sweep_figures(out_base, case; rho = sqrt(sm["rho_sq"]))
            @test rescaled.delta_min ≈ sm["delta_min"] rtol = 1e-12
        end
        figs_refit = sweep_figures(out_base, "mini_sweep"; refit = true)
        @test figs_refit.suffix == ""
        @test hasproperty(figs_refit, :slope_window) && hasproperty(figs_refit, :slope)
        sm_mini = TOML.parsefile(
            joinpath(out_base, "sweeps", "mini_sweep", "sweep_meta.toml"),
        )
        @test haskey(sm_mini, "slope_window") && haskey(sm_mini, "slope_window_err")
        # persisted (or recomputed) window exponent agrees with the refit
        if sm_mini["slope_window"] >= 0
            @test figs_refit.slope_window ≈ sm_mini["slope_window"] rtol = 1e-9
        end
        @test isfinite(figs.slope) # figs = sweep_figures(out_base, "mini_sweep") above
        # refit replot child: rewrites the scaling figure in place (backups kept)
        cmd_refit = addenv(`$jlbin --startup-file=no $replot_script $out_base --refit`,
            "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
        run(pipeline(cmd_refit, stdout = devnull, stderr = devnull))
        @test isfile(joinpath(out_base, "sweeps", "mini_sweep", "scaling_plot.png"))

        # panel loaders and layout-driven composites from the persisted tables
        pd = sweep_panel_data(out_base, "mini_sweep")
        @test length(pd.deltas) == length(pd.d2_num) == length(pd.clean)
        @test isfinite(pd.delta_min)
        zd = zone_panel_data(out_base, "mini_spin_map")
        @test length(zd.x) == length(zd.y) == length(zd.prior_limited)
        mktempdir() do tmp
            layout_path = joinpath(tmp, "layout.toml")
            write(
                layout_path,
                """
[[figures]]
name = "mini_scaling"
kind = "scaling"
ncols = 1
[[figures.panels]]
run = '$(out_base)'
case = "mini_sweep"
[[figures.panels]]
run = '$(out_base)'
case = "mini_unequal"
[[figures]]
name = "mini_zones"
kind = "zone"
labels = ["(a)", "(b)"]
[[figures.panels]]
run = '$(out_base)'
case = "mini_mass_time"
[[figures.panels]]
run = '$(out_base)'
case = "mini_spin_map"
[[figures]]
name = "mini_residual"
kind = "residual"
[[figures.panels]]
run = '$(out_base)'
case = "mini_sweep"
""",
            )
            figs_c = composite_figures(layout_path)
            @test length(figs_c) == 3
            @test [f.name for f in figs_c] ==
                  ["mini_scaling", "mini_zones", "mini_residual"]
            @test all(f -> f.figure isa CD.Plotting.Figure, figs_c)
            # panel labels only where the layout asks for them
            panel_labels(fig) = [
                c.text[] for c in fig.content if c isa CD.Plotting.Label &&
                    c.text[] in ("(a)", "(b)")
            ]
            @test isempty(panel_labels(figs_c[1].figure))
            @test panel_labels(figs_c[2].figure) == ["(a)", "(b)"]
            # a single-column stack is as wide as one panel; the export scale
            # is taken from that width
            @test canvas_width(figs_c[1].figure) == 600
            save_figure(figs_c[1].figure, joinpath(tmp, "composite");
                pt_per_unit = 400 / canvas_width(figs_c[1].figure))
            @test isfile(joinpath(tmp, "composite.pdf"))
            @test isfile(joinpath(tmp, "composite.png"))
            @test_throws ArgumentError save_figure(figs_c[1].figure,
                joinpath(tmp, "bad");
                pt_per_unit = 0.0)
            # layout validation: enumerated kind, no unknown keys
            bad_kind = joinpath(tmp, "bad_kind.toml")
            write(
                bad_kind,
                """
[[figures]]
name = "bad"
kind = "spectrum"
[[figures.panels]]
run = '$(out_base)'
case = "mini_sweep"
""",
            )
            @test_throws ArgumentError composite_figures(bad_kind)
            bad_key = joinpath(tmp, "bad_key.toml")
            write(
                bad_key,
                """
[[figures]]
name = "bad"
kind = "scaling"
foo = 1
[[figures.panels]]
run = '$(out_base)'
case = "mini_sweep"
""",
            )
            @test_throws ArgumentError composite_figures(bad_key)
        end
        rz = zone_map_figure(out_base, "mini_spin_map"; rho = 2.0)
        @test rz.suffix == "_rho2p0" && rz.contour isa DataFrame
        @test all(rz.contour.R_Capped .<= rz.contour.R_Box .+ 1e-12)
        @test all(isapprox.(rz.contour.R_Capped, cm2.R_Capped; rtol = 1e-12))

        # exit-status contract of the CLI: a configuration the pipeline cannot
        # honor ends with status 2 before any output exists
        pipeline_script = joinpath(dirname(@__DIR__), "scripts", "run_pipeline.jl")
        bad_cfg = joinpath(dir, "bad_key.toml")
        write(bad_cfg, "[hardware]\nrequire_gp = true\n")
        bad_out = joinpath(dir, "bad_out")
        cmd_bad = addenv(
            `$jlbin --startup-file=no $pipeline_script --config $bad_cfg --output-dir $bad_out`,
            "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
        proc_bad =
            run(pipeline(ignorestatus(cmd_bad), stdout = devnull, stderr = devnull))
        @test proc_bad.exitcode == 2
        @test !isdir(bad_out)
        stage_err = CD.PipelineStageError(out_base, ["mini_sweep", "mini_spin_map"])
        @test occursin("mini_sweep, mini_spin_map", sprint(showerror, stage_err))

        # collect_plots subprocess: flat browsing PNGs from a run path
        # selector plus a destination argument (same env scrubbing as the
        # replot child)
        collect_script = joinpath(dirname(@__DIR__), "scripts", "collect_plots.jl")
        browse_dir = joinpath(dir, "browse")
        cmd_collect =
            addenv(`$jlbin --startup-file=no $collect_script $browse_dir $out_base`,
                "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
        run(pipeline(cmd_collect, stdout = devnull, stderr = devnull))
        run_label = basename(out_base)
        for png in ("$(run_label)_sweep_mini_sweep_scaling.png",
            "$(run_label)_sweep_mini_sweep_residual.png",
            "$(run_label)_map_mini_spin_map.png")
            @test isfile(joinpath(browse_dir, png))
        end

        # rerun with the same config must not overwrite: suffixed run dir
        out2 = run_pipeline(cfg_path, dir, "outputs")
        @test out2 != out_base && isdir(out2)

        # an overlay (base_config + partial tables) runs end to end: the
        # merged [pipeline]/[mapping] values take effect, the snapshot is
        # self-contained and rehashes to the run's identifier, and the
        # metadata records both files
        overlay_path = joinpath(dir, "overlay.toml")
        write(
            overlay_path,
            """
base_config = "config.toml"
[pipeline]
run_1d_sweeps = false
[mapping]
n_angles = 16
max_refine_levels = 1
corner_bisect_iters = 0
""",
        )
        out3 = run_pipeline(overlay_path, dir, "outputs")
        @test startswith(basename(out3), run_id_from_config(overlay_path))
        snap = TOML.parsefile(joinpath(out3, "config.toml"))
        @test !haskey(snap, "base_config")
        @test snap["pipeline"]["run_1d_sweeps"] == false
        @test snap["pipeline"]["rng_seed"] == 11
        @test snap["mapping"]["n_angles"] == 16
        @test run_id_from_config(joinpath(out3, "config.toml")) ==
              run_id_from_config(overlay_path)
        meta3 = TOML.parsefile(joinpath(out3, "metadata.toml"))
        @test meta3["config_file"] == "overlay.toml"
        @test meta3["base_config"] == "config.toml"
        @test meta["config_file"] == "config.toml" && meta["base_config"] == ""
        @test isfile(joinpath(out3, "maps", "mini_spin_map", "confusion_contour.csv"))
        @test !isdir(joinpath(out3, "sweeps"))
    end
end
