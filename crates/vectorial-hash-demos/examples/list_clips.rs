//! List the animation clips in each horde/siege model — decides which clips
//! the demos can use (idle for the dormant carpet? a stand-up for waking?).
//! `cargo run -p vectorial-hash-demos --example list_clips`

fn main() {
    for name in ["zombie", "skeleton_a", "slime", "skeleton_sword", "bat", "anne", "pirate_captain", "sharky", "henry", "witch"] {
        let path = format!("crates/vectorial-hash-demos/assets/siege/models/{name}.glb");
        let bytes = std::fs::read(&path).expect("run from the workspace root");
        let (doc, _, _) = gltf::import_slice(&bytes).expect("glb");
        let clips: Vec<String> = doc.animations().map(|a| a.name().unwrap_or("?").to_string()).collect();
        println!("{name:>16}: {}", clips.join(" · "));
    }
}
