# Contributing

Issues and pull requests are welcome. The package is a research code with a
fixed scope, the quartic distinguishability law and the zones of confusion for
a LISA-like detector; the [roadmap](docs/src/roadmap.md) lists the extensions
I plan, and a proposal outside it is best raised as an issue first.

## Reporting a problem

Open an issue with the Julia version, the operating system, the GPU backend
if any (the `backend` line of the run's `metadata.toml`), the configuration
file and the relevant part of `run.log`. A run directory's `metadata.toml`,
`hardware.txt` and `Manifest.toml` identify the state that produced a result.

## Working on the code

```bash
julia -i activate.jl                                    # package environment
julia -e 'include("activate.jl"); using Pkg; Pkg.test()'
julia -e 'using JuliaFormatter; format(".")'            # .JuliaFormatter.toml
julia docs/make.jl                                      # strict documentation build
julia --threads=auto bench/run_benchmarks.jl
```

The test suite includes the static checks (Aqua, JET, ExplicitImports) and the
physics validation against the committed fixtures; a change of the model
regenerates them with `test/fixtures/generate_reference.jl` and says so in the
pull request. CI runs the suite on the declared floor (Julia 1.12) and on the
current stable release, the formatter check and the strict documentation build.

## Pull requests

- One change per pull request, with the tests and documentation it needs in
  the same commits.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `perf:`, `chore:`), with a body stating what changed
  and why.
- Public functions and structs carry a docstring; configuration keys are
  validated on load and documented in `docs/src/parameters.md`.
- A change that alters numerical results names the run identifiers it
  affects (the run id is the hash of the configuration).
- `Manifest.toml` is not tracked and stays out of pull requests.
- User-visible changes get an entry under `[Unreleased]` in `CHANGELOG.md`.
