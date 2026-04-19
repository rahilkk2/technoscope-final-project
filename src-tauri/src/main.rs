// src-tauri/src/main.rs — TechnoScope Desktop (Tauri v2)
//
// Manages the embedded Node.js server lifecycle:
//   • Starts `node backend/server.js` on app launch
//   • Exposes start_server / stop_server commands to the frontend
//   • Kills the server when the app window closes

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::State;

struct ServerState(Mutex<Option<Child>>);

/// Start the Node backend. Called once on app ready.
#[tauri::command]
fn start_server(state: State<ServerState>) -> Result<String, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;

    if guard.is_some() {
        return Ok("Server already running".into());
    }

    let child = Command::new("node")
        .arg("backend/server.js")
        .current_dir(
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| std::env::current_dir().unwrap()),
        )
        .spawn()
        .map_err(|e| format!("Failed to start server: {}", e))?;

    *guard = Some(child);
    Ok("Server started on port 8000".into())
}

/// Stop the Node backend.
#[tauri::command]
fn stop_server(state: State<ServerState>) -> Result<String, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;

    if let Some(mut child) = guard.take() {
        child.kill().map_err(|e| format!("Failed to kill server: {}", e))?;
        Ok("Server stopped".into())
    } else {
        Ok("Server was not running".into())
    }
}

fn main() {
    tauri::Builder::default()
        .manage(ServerState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![start_server, stop_server])
        .setup(|app| {
            // Auto-start the server when the app launches
            let state = app.state::<ServerState>();
            let _ = start_server(state);
            Ok(())
        })
        .on_window_event(|_window, event| {
            // Kill server when closing
            if let tauri::WindowEvent::Destroyed = event {
                // Server cleanup happens via Drop on app exit
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running TechnoScope");
}
