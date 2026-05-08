# IEEE-Style Final Report Package

This folder contains a concise two-column IEEE-style final report for the LLM course project.

Files:

- `final_report.tex`: primary IEEE-style LaTeX source.
- `final_report.md`: readable Markdown companion with the same substance.
- `final_report_print.html`: standalone two-column browser-printable version.
- `assets/`: clean report-specific PNG figures used by the report.
- `build_report_figures.ps1`: regenerates the report figures from the final CSV tables.

Group member:

- Mehmet Can Ozen, Student ID 20210808020.

## How To Compile

Recommended: upload this entire folder to Overleaf and compile `final_report.tex`.

The local machine currently does not expose `pdflatex`, `xelatex`, or `tectonic` on the command path, so the PDF was not built locally by this package. The source is self-contained except for the PNG files in `assets/`.

To regenerate the figures after rerunning experiments:

```powershell
cd C:\SoftwareProjects\LLMTermProject
powershell -ExecutionPolicy Bypass -File .\docs\submission\final_report_ieee\build_report_figures.ps1
```

If you install a LaTeX engine locally, run:

```powershell
cd C:\SoftwareProjects\LLMTermProject\docs\submission\final_report_ieee
pdflatex final_report.tex
pdflatex final_report.tex
```

No-LaTeX fallback:

1. Open `final_report_print.html` in Chrome or Edge.
2. Press `Ctrl+P`.
3. Destination: `Save as PDF`.
4. Layout: portrait.
5. Margins: default or none, whichever keeps it within 2-3 pages.
6. Enable background graphics only if the figures look washed out.

## Before Submission

Open the compiled PDF and check:

- It is 2-3 pages or close to the instructor limit.
- The name and student ID are visible under the title.
- Tables are readable.
- Figures render correctly.
- The limitations are included.
- The GitHub link appears in the conclusion or footnote.
