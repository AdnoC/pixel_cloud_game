//! This example demonstrates the built-in 3d shapes in Bevy.
//! The scene includes a patterned texture and a rotation for visualizing the normals and UVs.

use rand::prelude::*;
use rand_distr::StandardNormal;

use std::{
    collections::hash_map::Entry,
    f32::consts::{PI, TAU},
};

use bevy::{
    // render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    render::camera::ScalingMode, window::CursorGrabMode, input::mouse::MouseMotion,
};
use image::{io::Reader as ImageReader, Rgba, RgbaImage};
use palette::{IntoColor, Srgb};

#[derive(Default, Resource)]
struct CurrentImage(RgbaImage);

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (1080.0, 1080.0).into(),
                        title: "Pixel Cloud".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (mouse_grab, mouse_input, rotate))
        .run();
}

/// A marker component for our shapes so we can query them separately from the ground plane
#[derive(Component)]
struct Shape;

#[derive(Component)]
struct MyParent;

#[derive(Component)]
struct MyCamera;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut rng = rand::thread_rng();

    // let img = ImageReader::open("zelda-box.jpg")
    let img = ImageReader::open("goomba.png")
        .expect("could not find image")
        .decode()
        .expect("could not decode image")
        .to_rgba8();
    let height = img.height();
    let width = img.width();
    const RADIUS: f32 = 0.55;
    let max_distance = u32::max(height, width) as f32;

    let parent = commands.spawn((MyParent, SpatialBundle::default())).id();
    let mut mats: std::collections::HashMap<Rgba<u8>, Handle<StandardMaterial>> =
        std::collections::HashMap::new();
    let sphere_mesh = meshes.add(Mesh::from(shape::UVSphere {
        radius: RADIUS,
        ..default()
    }));
    let mut children = Vec::new();
    for (px, py, pixel) in img.enumerate_pixels() {
        if pixel[3] < 128 {
            continue;
        }

        let material = match mats.entry(*pixel) {
            Entry::Occupied(o) => o.get().clone_weak(),
            Entry::Vacant(v) => {
                let rgb = [pixel[0], pixel[1], pixel[2]];
                let color = palette::cast::from_array::<Srgb<u8>>(rgb).into_linear::<f32>();

                v.insert(materials.add(StandardMaterial {
                    base_color: Color::Rgba {
                        red: color.red,
                        blue: color.blue,
                        green: color.green,
                        alpha: 1.0,
                    },
                    ..default()
                }))
                .clone()
            }
        };

        let x = px as f32 - (width as f32 / 2.0);
        let y = py as f32 - (height as f32 / 2.0);
        let max_radius_distance = (height as f32 / 2.0).powi(2) + (width as f32 / 2.0).powi(2);
        let max_z = (max_radius_distance - x.powi(2) - y.powi(2)).sqrt();

        let shape = commands.spawn((
            PbrBundle {
                mesh: sphere_mesh.clone(),
                material,
                transform: Transform::from_xyz(
                    x,
                    y,
                    if max_z == 0.0 {
                        0.0
                    } else {
                        rng.gen_range(-max_z..=max_z)
                    },
                ),
                ..default()
            },
            Shape,
        )).id();
        children.push(shape);
    }

    commands.entity(parent).push_children(&children);

    // commands.insert_resource(CurrentImage(img));

    let (cam_x, cam_y, cam_z) = {
        // https://math.stackexchange.com/a/1585996
        let x: f32 = rng.sample(StandardNormal);
        let y: f32 = rng.sample(StandardNormal);
        let z: f32 = rng.sample(StandardNormal);

        let norm = 1.0 / (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
        (x * norm * max_distance, y * norm * max_distance, z * norm * max_distance)
    };


    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 9000.0,
            range: 100.,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(cam_x, cam_y, cam_z).looking_at(Vec3::new(0., 0., 0.), -Vec3::Y),
        ..default()
    });

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, -(max_distance + 1.0))
                .looking_at(Vec3::new(0., 0., 0.), -Vec3::Y),
            projection: Projection::Orthographic(OrthographicProjection {
                scale: 1.0,
                scaling_mode: ScalingMode::Fixed {
                    width: width as f32 * 2.,
                    height: height as f32 * 2.,
                },
                ..default()
            }),
            ..default()
        },
        MyCamera,
    ));
}

fn rotate(mut query: Query<&mut Transform, With<MyParent>>, time: Res<Time>) {
    // for mut transform in &mut query {
    //     transform.rotate_y(time.delta_seconds());
    // }
}

fn mouse_grab(mouse_button_input: Res<Input<MouseButton>>, mut windows: Query<&mut Window>) {
    let mut window = windows.single_mut();
    if mouse_button_input.just_pressed(MouseButton::Left) {
        window.cursor.visible = false;
        window.cursor.grab_mode = CursorGrabMode::Locked;
    }
    if mouse_button_input.just_released(MouseButton::Left) {
        window.cursor.visible = true;
        window.cursor.grab_mode = CursorGrabMode::None;
    }
}

fn mouse_input(mouse_button_input: Res<Input<MouseButton>>, mut mouse_motion_events: EventReader<MouseMotion>, mut query: Query<&mut Transform, With<MyParent>>, time: Res<Time>) {
    if mouse_button_input.pressed(MouseButton::Left) {
        for event in mouse_motion_events.iter() {
            let dx = event.delta.y;
            let dy = -event.delta.x;
            let mut transform = query.single_mut();
            transform.rotate_x(dx * 0.1 * time.delta_seconds());
            transform.rotate_y(dy * 0.1 * time.delta_seconds());

            // transform.translate_around(
            //     Vec3::new(0., 0., 0.),
            //     Quat::from_rotation_y(dx * 0.1 * time.delta_seconds()),
            //     );
            // transform.translate_around(
            //     Vec3::new(0., 0., 0.),
            //     Quat::from_rotation_x(dy * 0.1 * time.delta_seconds()),
            //     );
            // transform.look_at(Vec3::new(0., 0., 0.), -Vec3::Y);
        }
    }
}
