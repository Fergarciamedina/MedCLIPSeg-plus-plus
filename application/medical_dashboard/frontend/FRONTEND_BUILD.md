# Dashboard frontend build

Location `application/medical_dashboard/frontend/` · [Up to medical_dashboard/](../DASHBOARD_GUIDE.md)

This folder defines the browser build. Follow the [UI source guide](src/DASHBOARD_UI.md) for the components that turn an API response into a visible result.

| File or folder | Role |
| --- | --- |
| [index.html](index.html) | Browser entry point. Mounts the React view in the page. |
| [package.json](package.json) · [package-lock.json](package-lock.json) | Define development and build scripts, direct dependencies and the locked npm dependency tree. |
| [vite.config.ts](vite.config.ts) | Adds React support and proxies development requests under `/api` to the local FastAPI server. |
| [tsconfig.json](tsconfig.json) · [tsconfig.app.json](tsconfig.app.json) · [tsconfig.node.json](tsconfig.node.json) | Set the TypeScript project references and compiler options for browser sources and build tooling. |
| [postcss.config.js](postcss.config.js) · [tailwind.config.js](tailwind.config.js) | Configure CSS processing, source scanning, theme extension and vendor prefixes. |
| [src/](src/DASHBOARD_UI.md) | The React workflow and styles. |

Use the [dashboard guide](../DASHBOARD_GUIDE.md) to start both processes or verify the frontend build.
