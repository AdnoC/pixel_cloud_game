cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --out-name cloud_pixel_game --out-dir target/wasm_js --target web target/wasm32-unknown-unknown/release/pixel_cloud_game.wasm
