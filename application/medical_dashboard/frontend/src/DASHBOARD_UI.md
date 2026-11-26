# Dashboard UI source

Location `application/medical_dashboard/frontend/src/` · [Up to frontend/](../FRONTEND_BUILD.md)

These files implement the browser experience for submitting scans, viewing their stored segmentation overlays and review status, and browsing eligible reference matches.

| File | Role |
| --- | --- |
| [main.tsx](main.tsx) | Mounts the React app, manages selected uploads and request state, sends API requests, and renders the image, status and reference result panels. |
| [style.css](style.css) | Shared page layout, typography, controls and responsive positioning. |
| [status.css](status.css) | Styling for evaluation states and related result elements. |
| [palette.css](palette.css) | Color palette and visual overrides for controls and panels. |

[Frontend build](../FRONTEND_BUILD.md) · [Response fields](../../../stages/stage6_system_output/DASHBOARD_OUTPUTS.md)
