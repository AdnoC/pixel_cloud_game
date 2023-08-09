.\build_wasm.bat
git branch -D web
git checkout --orphan web
git reset
git add --force index.html target\wasm_js\* assets\* goomba.png
git commit -m "Stage for web"
git push -u origin web --force
git checkout master --force
