# Mac Studio setup — 29 September 2026

Read alongside [HANDOFF.md](HANDOFF.md); this records the new machine and does
not amend historical protocols. Host: macOS 26.5.1, Apple Silicon (`arm64`).

## Blender

Installed Blender **5.2.2 LTS**, build `d13f752e3b9c`, matching the version reported
in the handoff. Official installer:

https://download.blender.org/release/Blender5.2/blender-5.2.2-macos-arm64.dmg

The downloaded installer matched the macOS ARM64 entry in the official
`blender-5.2.2.sha256` file:

```text
dc4125399b8bfefe283cc1624d6cfc7809d1cac20ace51072127eb371f31f210
```

App: `/Applications/Blender.app`. Terminal shortcut:
`/Users/hinesa/.local/bin/blender` is a shell launcher that executes
`/Applications/Blender.app/Contents/MacOS/Blender`; that directory was already on
PATH. A subsequent script smoke check found that a simple symlink could print
the version but failed to locate Blender's bundled Python; it was replaced with
the launcher on 30 September. `/usr/local/bin` did not permit a non-administrator write. The installer
volume was ejected after copying the app.

Verification commands:

```sh
blender --version
blender --background --factory-startup --python-expr 'import bpy, platform; print("APIAVIZ_CHECK", bpy.app.version_string, platform.machine()); prefs = bpy.context.preferences.addons["cycles"].preferences; print("CYCLES_BACKENDS", prefs.get_device_types(bpy.context))'
```

Background startup passed outside the agent sandbox and reported `arm64` and
the Metal Cycles backend. Inside the sandbox, startup exited with signal 11
during Metal hardware detection. Future agent Blender launches may require
execution outside the sandbox. No render was performed, GPU performance was
not measured, and no renderer preferences were saved.

## Project environment and checks

Pixi and `.pixi/envs/default` were already present. Both Pixi and the environment's
Python executable are ARM64. These commands used the existing locked environment:

```sh
pixi run --locked python -c 'import sys, platform, numpy, torch; print(sys.executable); print(platform.machine()); print("NumPy", numpy.__version__); print("PyTorch", torch.__version__)'
pixi run --locked test > /private/tmp/apiaviz-studio-tests.log 2>&1
```

Imports passed: NumPy 1.26.4, PyTorch 2.12.1. The unittest suite ran 102 tests:
101 passed and one errored. The error was
`test_matrix_has_all_378_unique_conditions_and_both_phases`, which requires the
missing local artifact `apiaviz/output/controller-full/protocol.json`.
Neither `apiaviz/output/` nor `apiaviz/models/` was present during inspection.
Restore required artifacts with provenance before attempting historical replay.

Pixi warned about the deprecated CUDA `system-requirements` syntax and the older
v6 lockfile format. One sandboxed invocation also misreported the host archspec
as x86_64; executable inspection and Python both confirmed ARM64. No manifest,
lockfile, physics, controller, frozen protocol or experiment state was changed.
The cancelled navigation run and paused controller study were not started.
