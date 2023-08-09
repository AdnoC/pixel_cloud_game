// The meat of this taken from https://github.com/BorisBoutillier/Kataster
use bevy::{prelude::*, render::render_resource::AsBindGroup, reflect::{TypeUuid, TypePath}};

pub struct BackgroundPlugin;
impl Plugin for BackgroundPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<BackgroundMaterial>::default())
            .add_systems(Startup, spawn_background)
            .add_systems(Update, update_background_time);
    }
}
const WIDTH: f32 = 100.0;
const HEIGHT: f32 = 100.0;
// Spawn a simple stretched quad that will use of backgound shader
fn spawn_background(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
) {
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
        transform: Transform {
            translation: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(WIDTH, HEIGHT, 1.0),
            ..default()
        },
        material: materials.add(BackgroundMaterial { time: 0.0 }),
        ..default()
    });
}

#[derive(AsBindGroup, Debug, Clone, TypeUuid, TypePath)]
#[uuid = "cdbb9a45-07ca-43c3-9943-57321f353a8b"]
struct BackgroundMaterial {
    #[uniform(0)]
    time: f32,
}

impl Material for BackgroundMaterial {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        "background.wgsl".into()
    }
}
fn update_background_time(
    time: Res<Time>,
    // state: Res<State<AppState>>,
    mut backgrounds: ResMut<Assets<BackgroundMaterial>>,
) {
    // if state.get() != &AppState::GamePaused {
        for (_, background) in backgrounds.iter_mut() {
            background.time += time.delta_seconds();
        }
    // }
}
