# Resource planning, seeding and the map geometry of the driver. Included by runtests.jl.

@testset "Orchestrator: resource planning, seeding, mirroring" begin
    # an out-of-memory or interrupt raised inside a work-item task reaches
    # the stage guard wrapped by the task and by @sync; it stays fatal
    is_fatal = CD.Orchestrator.is_fatal
    @test is_fatal(OutOfMemoryError()) && is_fatal(InterruptException())
    @test !is_fatal(ErrorException("stage failure"))
    wrapped = try
        CD.Orchestrator.parallel_foreach(_ -> throw(OutOfMemoryError()), 2, 2)
    catch err
        err
    end
    @test wrapped isa CompositeException && is_fatal(wrapped)
    ordinary = try
        CD.Orchestrator.parallel_foreach(_ -> error("stage failure"), 2, 2)
    catch err
        err
    end
    @test ordinary isa CompositeException && !is_fatal(ordinary)
    mktempdir() do dir
        path = joinpath(dir, "c.toml")
        grid = "[grid]\nT_obs = 1.0e5\nf_min = 1.0e-3\nf_max = 2.0e-3\n"
        write(path, grid * "[safety]\nmax_ram_gb = 0.001\n")
        cfg = load_and_validate_config(path)
        @test_throws CD.Orchestrator.ResourceBudgetError CD.Orchestrator.plan_resources(
            cfg, 10^6, 2, CPU())
        # a budget holding exactly two tasks of a million-bin grid
        # downscales the concurrency (needs ≥ 3 threads to be observable)
        if Threads.nthreads() >= 3
            write(
                path,
                grid * "[safety]\nmax_ram_gb = 1.3\n[hardware]\nmax_threads = 8\n",
            )
            cfg2 = load_and_validate_config(path)
            sweep_tasks, map_tasks, est =
                @test_logs (:warn, r"Concurrency downscaled") CD.Orchestrator.plan_resources(
                    cfg2, 10^6, 2, CPU())
            @test sweep_tasks == 2 && map_tasks == 2 && est <= 1.3
        end
        # require_gpu aborts before the run directory exists whenever the
        # resolved backend is the CPU (no GPU package is loaded here)
        if get_best_backend(prefer = :auto) isa CPU
            write(
                path,
                grid * "[hardware]\ngpu_backend = \"auto\"\nrequire_gpu = true\n",
            )
            @test_throws ArgumentError run_pipeline(path, dir, "out")
            @test !isdir(joinpath(dir, "out"))
        end
    end

    # multi-start streams: a pure function of the work-item coordinates,
    # distinct across starts, stages and master seeds, and pinned at the
    # derived seed (first UInt64 of the SHA-256 of "seed|name|i|k")
    # rather than at a drawn value, so the pin depends on neither the
    # randn implementation nor the Xoshiro seeding of the Julia release
    # while a change of derivation still cannot silently alter the
    # perturbations recorded under one rng_seed
    stream(seed, name, k) = CD.Orchestrator.multi_start_stream(seed, name, 3, k)
    @test stream(42, "sweep", 2) == Xoshiro(0x810bd37a17606c49)
    @test randn(stream(42, "sweep", 2), 3) == randn(stream(42, "sweep", 2), 3)
    @test stream(42, "sweep", 2) != stream(42, "sweep", 3)
    @test stream(42, "sweep", 2) != stream(42, "map", 2)
    @test stream(42, "sweep", 2) != stream(43, "sweep", 2)

    # refinement compares neighbours on the circle. The neighbour of the last
    # direction of the half-circle is the mirror image of the first: with
    # radii 1, 1, 10, 10 at 0, π/2, π, 3π/2 the only jumps are π/2 → π and
    # 3π/2 → 2π, both folded onto the midpoint 3π/4
    constant(phis) = [(1.0, 1.0) for _ in phis]
    two = [(phi = 0.0, K = 1.0, g = 1.0, wall = 0x00),
        (phi = π / 2, K = 1.0, g = 1.0, wall = 0x00)]
    @test CD.ZoneSampling.refine_directions!(two, constant, e -> (1.0, 10.0);
        max_levels = 1, ratio_tol = 2.0) == 1
    @test [e.phi for e in two] ≈ [0.0, π / 2, 3π / 4]
    # a constant-curvature zone in the box (−2, 0.2) × (−1, 1): the capped radius
    # is continuous around the circle, |d ln r/dφ| ≤ 5 (at the corner where
    # tan φ = 5), so 64 directions on the half-circle (step π/64) stay below a
    # ratio of exp(5π/64) = 1.28 and nothing is added at a tolerance of 1.5
    function box_radius(phi)
        c, s = cos(phi), sin(phi)
        tx = c > 1e-14 ? 0.2 / c : c < -1e-14 ? -2.0 / c : Inf
        ty = s > 1e-14 ? 1.0 / s : s < -1e-14 ? -1.0 / s : Inf
        return min(tx, ty)
    end
    ring = [(phi = (k - 1) * π / 64, K = 1.0, g = 1.0, wall = 0x00) for k in 1:64]
    @test CD.ZoneSampling.refine_directions!(ring, constant,
        e -> (box_radius(e.phi), box_radius(e.phi + π)); max_levels = 6,
        ratio_tol = 1.5) == 0
    # the jump across the wrap from 3.0 to 0.5 + π is bisected at (3.5 + π)/2,
    # the direction pair of angle (3.5 - π)/2 on the half-circle
    wide = [(phi = 0.5, K = 1.0, g = 1.0, wall = 0x00),
        (phi = 3.0, K = 1.0, g = 1.0, wall = 0x00)]
    @test CD.ZoneSampling.refine_directions!(wide, constant, e -> (1.0, 10.0);
        max_levels = 1, ratio_tol = 2.0) == 1
    @test [e.phi for e in wide] ≈ [(3.5 - π) / 2, 0.5, 3.0]

    # mirroring: K and g are even, r_box is re-evaluated on the negated
    # direction (the box is asymmetric), degenerate and capped flags follow
    entries = [(phi = 0.0, K = 16.0, g = 1.0, wall = 0x00),
        (phi = π / 2, K = 1e-320, g = 1e-9, wall = 0x00)]
    box = (-1.5, 2.0, -0.5, 0.5)
    r_math_of(K) = K > CD.Geometry.K_UNDERFLOW ? (16.0 / K)^(1 / 4) : Inf
    polar = CD.ZoneSampling.mirror_to_full_circle(entries, box, r_math_of, 1e-6)
    @test polar.angle ≈ [0.0, π / 2, π, 3π / 2]
    @test polar.K_raw == [16.0, 1e-320, 16.0, 1e-320]
    @test polar.r_math[1] == polar.r_math[3] == 1.0 && isinf(polar.r_math[2])
    @test polar.r_box ≈ [2.0, 0.5, 1.5, 0.5]
    @test polar.r_cap ≈ [1.0, 0.5, 1.0, 0.5]
    @test polar.prior_limited == [false, true, false, true]
    @test polar.degenerate == [false, true, false, true]
    # a crossover vertex marked on one side only: that side is snapped onto
    # the wall and flagged although r_math sits a hair inside it; the
    # antipode keeps the plain comparison
    entries_w = [(phi = 0.0, K = 16.0 / (2.0 * (1 - 1e-7))^4, g = 1.0,
        wall = CD.ZoneSampling.WALL_SELF)]
    polar_w = CD.ZoneSampling.mirror_to_full_circle(entries_w, box, r_math_of, 1e-6)
    @test polar_w.r_cap[1] == 2.0 && polar_w.prior_limited[1]
    @test polar_w.r_cap[2] ≈ 1.5 && polar_w.prior_limited[2] # r_math ≈ 2 > 1.5 anyway
    # Fisher-normalized sampling: the direction map turns the sampling
    # angle into a parameter-space unit vector, and the mirrored half
    # uses its exact negation
    aniso = phi -> CD.ZoneSampling.unit_direction(phi, 1e-3, 1.0)
    c45, s45 = aniso(π / 4)
    @test hypot(c45, s45) ≈ 1.0 && s45 / c45 ≈ 1e3
    @test all(aniso(π / 4 + π) .≈ (-c45, -s45))
    polar_a = CD.ZoneSampling.mirror_to_full_circle(
        [(phi = π / 4, K = 16.0, g = 1.0, wall = 0x00)], box, r_math_of, 1e-6;
        direction = aniso)
    @test polar_a.dir_cos ≈ [c45, -c45] && polar_a.dir_sin ≈ [s45, -s45]
    @test polar_a.r_box ≈ [0.5 / s45, 0.5 / s45] # the ±y walls at ±0.5 dominate
    @test CD.ZoneSampling.wall_bit(0.5) == CD.ZoneSampling.WALL_SELF
    @test CD.ZoneSampling.wall_bit(4.0) == CD.ZoneSampling.WALL_ANTIPODE
    @test CD.ZoneSampling.wall_bit(-0.5) == CD.ZoneSampling.WALL_ANTIPODE

    # box-corner insertion: a ray through a finite box corner that is capped
    # there becomes a boundary vertex; the four corners fold onto two
    # half-circle directions
    entries_c = [(phi = 0.0, K = 1.0, g = 1.0, wall = 0x00),
        (phi = π / 2, K = 1.0, g = 1.0, wall = 0x00)]
    n_inserted = CD.ZoneSampling.insert_box_corner_vertices!(entries_c,
        phis -> [(1.0, 1.0) for _ in phis], (alpha, K) -> true,
        (-1.0, 1.0, -1.0, 1.0))
    @test n_inserted == 2 && length(entries_c) == 4
    @test any(e -> isapprox(e.phi, π / 4), entries_c) &&
          any(e -> isapprox(e.phi, 3π / 4), entries_c)
    @test issorted([e.phi for e in entries_c])
    # a symmetric box folds each corner pair onto one direction carrying
    # both wall bits (the direction and its antipode end on a corner)
    @test all(e -> e.wall == 0x03, filter(e -> !(e.phi in (0.0, π / 2)), entries_c))
    @test all(e -> e.wall == 0x00, filter(e -> e.phi in (0.0, π / 2), entries_c))
    # an uncapped corner inserts nothing
    @test CD.ZoneSampling.insert_box_corner_vertices!(copy(entries_c),
        phis -> [(1.0, 1.0) for _ in phis], (alpha, K) -> false, (-1.0, 1.0, -1.0, 1.0),
    ) == 0

    # boundary runs: contiguous same-class edges join into NaN-separated
    # polylines; the wrap-around edge starts a new run
    xs, ys = CD.Plotting.boundary_runs([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0],
        [true, true, false, true], true)
    @test isequal(xs, [0.0, 1.0, 2.0, NaN, 3.0, 0.0])
    @test isequal(ys, [0.0, 0.0, 1.0, NaN, 1.0, 0.0])
end
