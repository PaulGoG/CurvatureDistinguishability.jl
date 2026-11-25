# Static quality assurance: Aqua, ExplicitImports, JET. Included by runtests.jl.

@testset "Static QA" begin
    # the persistent-tasks probe spawns a child precompilation whose
    # completion marker is unreliable on cold macOS/Windows runners and under
    # coverage instrumentation (the instrumented child precompile of the
    # full dependency stack outruns the probe); the verdict is
    # platform-independent, so the probe runs on uninstrumented Linux only
    run_probe = Sys.islinux() && Base.JLOptions().code_coverage == 0
    Aqua.test_all(CurvatureDistinguishability; persistent_tasks = run_probe)
    # the qualified-access publicity check is deliberately not enforced:
    # ForwardDiff and Optim expose their documented API (Dual, value,
    # partials, minimizer, …) without `public` annotations
    @test ExplicitImports.check_no_implicit_imports(CurvatureDistinguishability) ===
          nothing
    @test ExplicitImports.check_no_stale_explicit_imports(
        CurvatureDistinguishability,
    ) === nothing
    # the top-module check does not recurse into submodules; each is
    # checked on its own
    for submodule in (CD.Physics, CD.Detector, CD.Residuals, CD.Bounds,
        CD.Geometry, CD.Fitting, CD.Inference, CD.Backends, CD.Config,
        CD.Provenance, CD.Plotting, CD.ZoneSampling, CD.Orchestrator, CD.RunFigures,
        CD.Heartbeat, CD.Supervision, CD.Checkpoint, CD.Campaign)
        @test ExplicitImports.check_no_stale_explicit_imports(submodule) ===
              nothing
    end
    @test ExplicitImports.check_all_explicit_imports_via_owners(
        CurvatureDistinguishability,
    ) === nothing

    # JET package analysis with error points restricted to this package's
    # modules (upstream abstract-interpretation artifacts in
    # Base/Optim/Makie internals are out of scope). Under
    # KernelAbstractions 0.9 the analysis can report, as a method error, the
    # GPU branch of the backend union split at the deliberately untyped
    # device-buffer barrier (launch_loss!/launch_loss_lanes!/column_sums_via!,
    # one per kernel: loss_bins!, loss_bins_lanes!, partial_column_sums!): the
    # GPU kernel-call method exists only once a GPU backend package is loaded,
    # while that branch is unreachable by construction (get_best_backend
    # returns only registered backends). Whether these artifacts appear
    # depends on the resolved environment (three on a fresh resolution, none
    # on the lowest one of CI). From 0.10 the CPU backend launches through the
    # same kernel interface as the GPU backends and none remains. Any other
    # report fails.
    # JET tracks compiler internals and routinely breaks on pre-release
    # Julia, so the analysis gates stable releases only
    if isempty(VERSION.prerelease)
        jet_modules = (CD, CD.Physics, CD.Detector, CD.Residuals, CD.Bounds,
            CD.Geometry, CD.Fitting, CD.Inference, CD.Backends, CD.Config,
            CD.Provenance, CD.Plotting, CD.ZoneSampling, CD.Orchestrator, CD.RunFigures,
            CD.Heartbeat, CD.Supervision, CD.Checkpoint, CD.Campaign)
        jet = JET.report_package(CD; target_modules = jet_modules,
            toplevel_logger = nothing)
        reports = JET.get_reports(jet)
        barrier = (:launch_loss!, :launch_loss_lanes!, :column_sums_via!)
        function barrier_site(r)
            i = findfirst(f -> f.linfo.def isa Method && f.linfo.def.name in barrier, r.vst)
            return i === nothing ? nothing : r.vst[i].linfo.def.name
        end
        sites = barrier_site.(reports)
        if pkgversion(KernelAbstractions) < v"0.10"
            # only the method-error artifacts, at most one per barrier function
            @test all(r -> r isa JET.MethodErrorReport, reports)
            @test !any(isnothing, sites) && allunique(sites)
        else
            @test isempty(reports)
        end
    end
end
