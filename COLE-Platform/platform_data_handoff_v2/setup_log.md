# Real environment smoke setup log

## Host and environment

- Operating system: Windows 10 build 22621
- Conda environment: `cole-platform`
- Python: 3.7.1 (conda-forge)
- Repository working directory: `C:\Users\admin\Desktop\zzz\COLE-Platform-main\COLE-Platform-main`

The environment is isolated from the base Python installation. No COLE model
was loaded and no training command was run.

## Installed smoke dependencies

| Package | Version | Source |
|---|---:|---|
| gym | 0.21.0 | PyPI |
| numpy | 1.21.4 | conda-forge |
| scipy | 1.7.3 | conda-forge |
| torch | 1.12.1 | PyPI |
| stable-baselines3 | 1.7.0 | PyPI |
| importlib-metadata | 4.13.0 | PyPI |
| tqdm | 4.67.1 | conda-forge |
| setuptools | 65.5.0 | PyPI |
| wheel | 0.38.4 | PyPI |

## Setup decisions and first errors

1. Default Anaconda channels were not used because their Terms of Service had
   not been accepted. The environment was created with conda-forge override.
2. `scipy==1.5.4` was unavailable for this Windows/Python 3.7 conda-forge
   solve. Conda selected SciPy 1.7.3; the real environment smoke passed.
3. The first Gym installation failed because Gym 0.21 metadata is incompatible
   with setuptools 69. Installing setuptools 65.5.0 and wheel 0.38.4 allowed
   the repository-pinned Gym 0.21.0 to build.
4. Gym 0.21 failed with importlib-metadata 6.7 because it expects the older
   `entry_points().get(...)` API. The repository-pinned
   `importlib-metadata==4.13.0` fixed the import.
5. Directly invoking the environment Python omitted Conda DLL activation and
   caused a NumPy DLL error. All verified commands use `conda run`.
6. The repository mixes imports from `overcooked_ai_py` and
   `overcooked_ai.overcooked_ai_py`; both local parent paths are included in
   `PYTHONPATH` for the smoke command without modifying source code.

## Exact smoke command

```powershell
$root=(Resolve-Path '.').Path
$env:PYTHONPATH=($root + ';' + (Resolve-Path 'overcookedgym\human_aware_rl').Path + ';' + (Resolve-Path 'overcookedgym\human_aware_rl\overcooked_ai').Path)
$env:MPLCONFIGDIR=(Join-Path $root 'platform_data_handoff_v2\logs\matplotlib-cache')
& 'C:\Users\admin\miniconda3\Scripts\conda.exe' run --name cole-platform --no-capture-output python overcookedgym\scripts\real_env_smoke.py --output-dir platform_data_handoff_v2 --layout-id simple --seed 1 --episode-id real-env-smoke-001
```

## Result

- Exit code: 0
- Layout: `simple`
- Seed: 1
- Joint action on every step: `[4, 4]` (`stay`, `stay`)
- Steps: 400
- Wrapped/unwrapped comparison: PASS at every step
- Final state comparison: PASS
- v2 schema validation: PASS
- Events emitted: 0, as expected for an all-stay episode

## Known environment warning

After pip downgraded setuptools and wheel, `conda list` warned that two older
Conda metadata JSON files could not be removed. Runtime imports report the
intended versions (setuptools 65.5.0 and wheel 0.38.4), and all smoke/tests
pass. The internal Conda metadata files were not manually deleted.
