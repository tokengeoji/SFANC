# DSP ANC firmware

Both project folders in this directory are importable TI Code Composer Studio projects for the
LCDK OMAP-L138/TMS320C6748 target.

- `FxNLMS0`: baseline filtered-x normalized LMS controller.
- `FxNLMS0_16_sfanc`: 16 kHz SFANC controller with six pre-trained filters and
  UART-driven filter selection from the PC SFANC runtime.

The shared-looking `lib/` folders remain inside each project intentionally:
Code Composer Studio owns their source discovery and include paths independently.
Build folders (`Debug/`, `Release/`), XDC-generated `.config/` and `src/`
folders, launch files, and backups are ignored because they contain generated or
machine-specific paths. Rebuild them locally after importing a project.
