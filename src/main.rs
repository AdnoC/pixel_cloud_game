//! This example demonstrates the built-in 3d shapes in Bevy.
//! The scene includes a patterned texture and a rotation for visualizing the normals and UVs.

// A bunch of wgsl typedefs:
// https://github.com/bevyengine/bevy/issues/5561
use rand::prelude::*;
use rand_distr::StandardNormal;
use crossbeam_channel::{bounded, Sender, Receiver};
use rayon::prelude::*;

use std::{
    sync::{OnceLock},
    collections::hash_map::Entry,
    f32::consts::{PI, TAU}, path::Path,
};

use bevy::{
    core::{Pod, Zeroable},
    core_pipeline::core_3d::Transparent3d,
    // render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    input::mouse::MouseMotion,
    pbr::{MeshPipeline, MeshPipelineKey, MeshUniform, SetMeshBindGroup, SetMeshViewBindGroup},
    prelude::*,
    render::camera::ScalingMode,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{GpuBufferInfo, MeshVertexBufferLayout},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, RenderCommand, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::{ExtractedView, NoFrustumCulling},
        Render, RenderApp, RenderSet,
    },
    window::CursorGrabMode,
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiSettings};
use image::{io::Reader as ImageReader, Rgba, RgbaImage, load_from_memory, SubImage};
use std::io::Cursor;
use palette::{IntoColor, Srgb};
mod background;
use background::BackgroundPlugin;


#[derive(Default, Resource)]
struct CurrentImage(RgbaImage);

/// A marker component for our shapes so we can query them separately from the ground plane
#[derive(Component)]
struct Shape;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct InstanceData {
    position: Vec3,
    scale: f32,
    color: [f32; 4],
}

#[derive(Component)]
struct MyParent;

#[derive(Component)]
struct MyCamera;

#[derive(Event)]
struct FixTransparencyEvent(pub u32, pub u32);

#[derive(Event)]
struct WinStateEvent(pub bool);

#[derive(Resource)]
struct ImageData(RgbaImage, Option<String>);
impl ImageData {
    fn new(img: RgbaImage, name: String) -> ImageData {
        ImageData(Self::crop_transparency(img), Some(name))
    }

    fn new_unnamed(img: RgbaImage) -> ImageData {
        ImageData(Self::crop_transparency(img), None)
    }

    fn crop_transparency(img: RgbaImage) -> RgbaImage {
        let mut min_x = img.width();
        let mut min_y = img.height();
        let mut max_x = 0;
        let mut max_y = 0;
        for (px, py, pixel) in img.enumerate_pixels() {
            if pixel[3] == u8::MAX {
                if px < min_x {
                    min_x = px;
                }
                if px > max_x {
                    max_x = px;
                }
                if py < min_y {
                    min_y = py;
                }
                if py > max_y {
                    max_y = py;
                }
            }
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        let mut img = img;
        image::imageops::crop(&mut img, min_x, min_y, width, height).to_image()
    }
}

#[derive(Resource, Default, Eq, PartialEq)]
enum AppState {
    #[default]
    InGame,
    TransparencyFixer,
}

#[derive(Resource, Default)]
struct UiData {
    loaded: bool,
    img_handles: Option<[egui::TextureHandle; 6]>,
    img_handle: Option<egui::TextureHandle>,
}

#[derive(Resource, Deref)]
struct WasmReceiver(Receiver<(Vec<u8>, String)>);

#[derive(Resource)]
struct WinTimer(f64);

#[cfg(target_family = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    pub fn load_image(img: &[u8], name: String) {
        match crate::IMG_QUEUE.get() {
            Some(tx) => tx.send((img.to_vec(), name)).expect("error sending wasm img data"),
            None => return,
        }
    }
}

static IMG_QUEUE: OnceLock<Sender<(Vec<u8>, String)>> = OnceLock::new();


fn main() {
    let img = load_from_memory(include_bytes!("../goomba.png"))
        .expect("could not find image")
        .to_rgba8();

    let mut app = App::new();
        app.add_plugins((
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (1080.0, 1080.0).into(),
                        title: "Pixel Cloud".to_string(),
                        // Anchors WASM version to a canvas with id "bevy"
                        canvas: Some("#bevy".to_string()),
                        ..default()
                    }),
                    ..default()
                }),
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            CustomMaterialPlugin,
        ))
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, init_cloud)
        .add_event::<FixTransparencyEvent>()
        .insert_resource(ImageData(img, Some("goomba.png".to_string())))
        .insert_resource(WinTimer(0.0))
        .insert_resource(AppState::default())
        .insert_resource(UiData::default())
        // .add_systems(Startup, setup)
        .add_systems(Update, (transparency_fixer_ui, ui_open))
        .add_systems(Update, (load_external_level, load_dd_level, mouse_grab, mouse_input, rotate, win_check, apply_transparency_fix));

        //#[cfg(not(target_family = "wasm"))]
        {
            app.add_plugins(BackgroundPlugin);
        }
        app.run();
}

// fn run_if_ui_closed(ui_state: Res<UiState>)


fn setup(mut commands: Commands) {
    let (tx, rx) = bounded::<(Vec<u8>, String)>(10);
    commands.insert_resource(WasmReceiver(rx));
    IMG_QUEUE.set(tx).expect("could not initialize wasm image queue");
}

fn ui_setup(mut commands: Commands) {

    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };

    // This is the texture that will be rendered to.
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };

    // fill image.data with zeroes
    image.resize(size);

}

fn apply_transparency_fix(mut ev_fix_transparency: EventReader<FixTransparencyEvent>, mut img_data: ResMut<ImageData>) {
    for ev in ev_fix_transparency.iter() {
        let img = &mut img_data.0;
        let clear = Rgba([0, 0, 0, 0]);
        let color = img.get_pixel(ev.0, ev.1).clone();

        let width = img.width();
        let height = img.height();
        let mut stack = vec![(ev.0, ev.1)];
        while let Some((x, y)) = stack.pop() {
            let pix = img.get_pixel(x, y);
            if pix[3] > 0 && *pix == color {
                img.put_pixel(x, y, clear);
                if x > 0 {
                    stack.push((x - 1, y));
                }
                if y > 0 {
                    stack.push((x, y - 1));
                }
                if x < width - 1 {
                    stack.push((x + 1, y));
                }
                if y < height - 1 {
                    stack.push((x, y + 1));
                }
            }
        }
    }
}

fn image_to_egue_image_data(img: &RgbaImage) -> egui::ImageData {
    use egui::{ColorImage, Color32, ImageData::Color};

    let img_data = ColorImage {
        size: [img.width() as usize, img.height() as usize],
        pixels: img.pixels().map(|p| Color32::from_rgba_premultiplied(p[0], p[1], p[2], p[3])).collect(),
    };
    Color(img_data)
}

fn scale_image(src: &RgbaImage, scale: u32) -> RgbaImage {
    if scale == 1 {
        return src.clone();
    }

    let width = src.width() * scale;
    let height = src.height() * scale;
    let mut pixels = vec![image::Rgba([0, 0, 0, 0]); width as usize * height as usize];
    for (src_x, src_y, pixel) in src.enumerate_pixels() {
        let start_y = src_y * scale;
        for py in start_y..(start_y + scale) {
            let start = (py * width + src_x * scale) as usize;
            let end = start + scale as usize - 1;
            // println!("px.len = {}, ({}, {}) * ({:?}) => ({}..={}) {:?} = {:?}", pixels.len(), src_x, src_y, (src.width(), src.height()), start, end, (width, height), pixel);
            let _ = &pixels[start..=end].fill(*pixel);
        }
    }
    RgbaImage::from_vec(width, height, pixels.iter().flat_map(|p| p.0).collect()).expect("resized image not valid")
}

fn transparency_fixer_ui(ui_state: Res<AppState>, mut ui_data: ResMut<UiData>, img_data: Res<ImageData>, mut ev_fix_transparency: EventWriter<FixTransparencyEvent>, mut contexts: EguiContexts) {
    let ctx = contexts.ctx_mut();
    if img_data.is_changed() {
        ui_data.loaded = false;
    }

    if *ui_state != AppState::TransparencyFixer {
        return;
    }


    egui::TopBottomPanel::top("header").show(ctx, |ui| {
        ui.heading("Select pixel to start transparency fix");
    });

    let scale = {
        let avail = ctx.available_rect();
        let target_w = (avail.max.x - avail.min.x) as u32;
        let target_h = (avail.max.y - avail.min.y) as u32;

        let scale_w = target_w / img_data.0.width();
        let scale_h = target_h / img_data.0.height();
        scale_w.min(scale_h)
    };
    if !ui_data.loaded {
        ui_data.loaded = true;

        // let img = &img_data.0;
        // let handles = [1, 2, 4, 8, 16, 32]
        //     .map(|scale| image_to_egue_image_data(&scale_image(img, scale)))
        //     .map(|img| ctx.load_texture(
        //                     "current-image",
        //                     img,
        //                     Default::default(),
        //                ));
        // ui_data.img_handles = Some(handles);


        let img = if scale > 1 {
            image_to_egue_image_data(&scale_image(&img_data.0, scale))
        } else {
            image_to_egue_image_data(&img_data.0)

        };
        let img_handle = ctx.load_texture(
            "current-image",
            img,
            Default::default(),
            );
        ui_data.img_handle = Some(img_handle);
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        // TODO: Handle too-big images via scroll.
        // They don't work performance-wise but I want to do this.
        egui::ScrollArea::neither().show(ui, |ui| {
            if let Some(img_handle) = &ui_data.img_handle {
                // let base_img = &img_handles[0];
                // let size_avail = ui.available_size();
                // let img_size = base_img.size_vec2();
                // let dx = size_avail[0] as usize / img_size[0] as usize;
                // let dy = size_avail[1] as usize / img_size[1] as usize;
                // let scale = dx.min(dy);
                // let handle_idx = {
                //     if scale >= 32 {
                //         5
                //     } else if scale >= 16 {
                //         4
                //     } else if scale >= 8 {
                //         3
                //     } else if scale >= 4 {
                //         2
                //     } else if scale >= 2 {
                //         1
                //     } else {
                //         0
                //     }
                // };
                // let img_handle = &img_handles[handle_idx];

                let img = ui.add(egui::widgets::Image::new(
                        img_handle.id(),
                        img_handle.size_vec2(),
                        )).interact(egui::Sense::click());
                if img.clicked() {
                    if let Some(ipp) = img.interact_pointer_pos {
                        let x = ((ipp[0] - img.rect.min.x)) as u32 / scale as u32;
                        let y = ((ipp[1] - img.rect.min.y)) as u32 / scale as u32;
                        println!("{:?} - {:?} / {} = {:?}\t{:?}", ipp, img.rect.min, scale, (x, y), img.rect.max);
                        ev_fix_transparency.send(FixTransparencyEvent(x, y));
                    }
                }
            }
        });
    });

    // egui::CentralPanel::default().show(ctx, |ui| {
    //     if let Some(img_handle) = &ui_data.img_handle {
    //         // let base_img = &img_handles[0];
    //         // let size_avail = ui.available_size();
    //         // let img_size = base_img.size_vec2();
    //         // let dx = size_avail[0] as usize / img_size[0] as usize;
    //         // let dy = size_avail[1] as usize / img_size[1] as usize;
    //         // let scale = dx.min(dy);
    //         // let handle_idx = {
    //         //     if scale >= 32 {
    //         //         5
    //         //     } else if scale >= 16 {
    //         //         4
    //         //     } else if scale >= 8 {
    //         //         3
    //         //     } else if scale >= 4 {
    //         //         2
    //         //     } else if scale >= 2 {
    //         //         1
    //         //     } else {
    //         //         0
    //         //     }
    //         // };
    //         // let img_handle = &img_handles[handle_idx];
    //
    //         ui.with_layout(egui::Layout::centered_and_justified(egui::Direction::TopDown), |ui| {
    //             let img = ui.add(egui::widgets::Image::new(
    //                     img_handle.id(),
    //                     img_handle.size_vec2(),
    //                     )).interact(egui::Sense::click());
    //             if img.clicked() {
    //                 if let Some(ipp) = img.interact_pointer_pos {
    //                     let x = ((ipp[0] - img.rect.min.x)) as u32 / scale as u32;
    //                     let y = ((ipp[1] - img.rect.min.y)) as u32 / scale as u32;
    //                     println!("{:?} - {:?} / {} = {:?}\t{:?}", ipp, img.rect.min, scale, (x, y), img.rect.max);
    //                     ev_fix_transparency.send(FixTransparencyEvent(x, y));
    //                 }
    //             }
    //         });
    //     }
    // });
}

fn ui_open(mut ui_state: ResMut<AppState>, key_input: Res<Input<KeyCode>>) {
    if key_input.just_pressed(KeyCode::O) {
        *ui_state = match *ui_state {
            AppState::TransparencyFixer => AppState::InGame,
            _ => AppState::TransparencyFixer,
        };
    }
}

fn load_external_level(mut img_data: ResMut<ImageData>, receiver: Res<WasmReceiver>) {
    for from_external in receiver.try_iter() {
        let img = load_from_memory(&from_external.0)
            .expect("could not find image")
            .to_rgba8();
        let name = Path::new(&from_external.1).file_name().expect("WASM path is not a file").to_string_lossy().to_string();
        *img_data = ImageData::new(img, name);
    }
}

fn load_dd_level(mut img_data: ResMut<ImageData>, mut dnd_evr: EventReader<FileDragAndDrop>) {
    for ev in dnd_evr.iter() {
        if let FileDragAndDrop::DroppedFile { path_buf, .. } = ev {
            let name = path_buf.file_name().expect("DND path is not a file").to_string_lossy().to_string();
            println!("Dropped file with path: {:?}", path_buf);
            if let Some(filename) = path_buf.to_str() {

                let img = ImageReader::open(filename)
                    .expect("could not find image")
                    .decode()
                    .expect("could not decode image")
                    .to_rgba8();
                *img_data = ImageData::new(img, name);
            }
        }
    }
}

fn init_cloud(
    mut commands: Commands,
    img_data: Res<ImageData>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut bg_size: ResMut<background::BackgroundSize>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    preexisting: Query<Entity, Or<(With<MyParent>, With<MyCamera>)>>
) {
    if !img_data.is_changed() {
        return;
    }
    for entity in preexisting.iter() {
        commands.entity(entity).despawn_recursive();
    }

    let img = &img_data.0;

    let height = img.height();
    let width = img.width();
    const RADIUS: f32 = 0.55;
    let max_distance = u32::max(height, width) as f32;

    let parent_bundle = {
        let mut parent_bundle = SpatialBundle::default();

        // https://stackoverflow.com/a/44031492
        let mut rng = rand::thread_rng();
        let u: f32 = rng.gen_range((0.0)..=(1.0));
        let v: f32 = rng.gen_range((0.0)..=(1.0));
        let w: f32 = rng.gen_range((0.0)..=(1.0));

        let quat = Quat::from_xyzw(
             (1. - u).sqrt() * (TAU * v).sin(),
             (1. - u).sqrt() * (TAU * v).cos(),
             (u).sqrt() * (TAU * w).sin(),
             (u).sqrt() * (TAU * w).cos(),
        );


        // parent_bundle.transform.rotation = quat;

        parent_bundle
    };
    let parent = commands.spawn((MyParent, parent_bundle)).id();
    let sphere_mesh = meshes.add(Mesh::from(shape::UVSphere {
        radius: RADIUS,
        ..default()
    }));
    let sphere_data: Vec<InstanceData> = img
        .enumerate_pixels()
        .collect::<Vec<(u32, u32, &image::Rgba<u8>)>>()
        .par_iter()
        .filter(|(_, _, pixel)| pixel[3] > 128)
        .map(|(px, py, pixel)| {
            let mut rng = rand::thread_rng();
            let rgb = [pixel[0], pixel[1], pixel[2]];
            let color = palette::cast::from_array::<Srgb<u8>>(rgb).into_linear::<f32>();

            let x = *px as f32 - (width as f32 / 2.0);
            let y = *py as f32 - (height as f32 / 2.0);
            let max_radius_distance = (height as f32 / 2.0).powi(2) + (width as f32 / 2.0).powi(2);
            let max_z = (max_radius_distance - x.powi(2) - y.powi(2)).sqrt();

            let position = Vec3::new(
                x,
                y,
                if max_z == 0.0 {
                    0.0
                } else {
                    rng.gen_range(-max_z..=max_z)
                },
            );
            InstanceData {
                position,
                scale: 1.0,
                color: [color.red, color.green, color.blue, 1.0],
            }
        })
        .collect();

        // for dat in sphere_data {
        //     let material = materials.add(StandardMaterial {
        //         base_color: Color::Rgba {
        //             red: dat.color[0],
        //             blue: dat.color[1],
        //             green: dat.color[2],
        //             alpha: dat.color[3],
        //         },
        //         ..default()
        //     });
        //     commands.spawn((
        //             PbrBundle {
        //                 mesh: sphere_mesh.clone(),
        //                 material,
        //                 transform: Transform::from_xyz(
        //                     dat.position.x,
        //                     dat.position.y,
        //                     dat.position.z,
        //                     ),
        //                     ..default()
        //             },
        //             Shape,
        //             ));
        //
        // }
    let spheres_id = commands.spawn((
        sphere_mesh,
        SpatialBundle::INHERITED_IDENTITY,
        InstanceMaterialData(sphere_data),
        NoFrustumCulling,
    )).id();

    commands.entity(parent).push_children(&[spheres_id]);

    // {
    //     let red = materials.add(StandardMaterial {
    //         base_color: Color::Rgba {
    //             red: 1.,
    //             blue: 0.,
    //             green: 0.,
    //             alpha: 1.,
    //         },
    //         ..default()
    //     });
    //     let green = materials.add(StandardMaterial {
    //         base_color: Color::Rgba {
    //             red: 0.,
    //             blue: 0.,
    //             green: 1.,
    //             alpha: 1.,
    //         },
    //         ..default()
    //     });
    //     let blue = materials.add(StandardMaterial {
    //         base_color: Color::Rgba {
    //             red: 0.,
    //             blue: 1.,
    //             green: 0.,
    //             alpha: 1.,
    //         },
    //         ..default()
    //     });
    //
    //
    //     // commands.spawn((
    //     //     PbrBundle {
    //     //         mesh: meshes.add(Mesh::from(shape::Cylinder {
    //     //             radius: 5.,
    //     //             height: 100.,
    //     //             resolution: 16,
    //     //             segments: 2,
    //     //
    //     //         })),
    //     //         material: red,
    //     //         transform: Transform::default().looking_at(
    //     //             Vec3::new(1., 0., 0.,),
    //     //             Vec3::ZERO
    //     //         ),
    //     //         ..default()
    //     //     },
    //     //     Shape,
    //     // ));
    //     // commands.spawn((
    //     //     PbrBundle {
    //     //         mesh: meshes.add(Mesh::from(shape::Cylinder {
    //     //             radius: 5.,
    //     //             height: 100.,
    //     //             resolution: 16,
    //     //             segments: 2,
    //     //
    //     //         })),
    //     //         material: green,
    //     //         transform: Transform::default().looking_at(
    //     //             Vec3::new(0., 1., 0.,),
    //     //             Vec3::ZERO
    //     //         ),
    //     //         ..default()
    //     //     },
    //     //     Shape,
    //     // ));
    //     // commands.spawn((
    //     //     PbrBundle {
    //     //         mesh: meshes.add(Mesh::from(shape::Cylinder {
    //     //             radius: 5.,
    //     //             height: 100.,
    //     //             resolution: 16,
    //     //             segments: 2,
    //     //
    //     //         })),
    //     //         material: blue,
    //     //         transform: Transform::default().looking_at(
    //     //             Vec3::new(1., 0., 0.,),
    //     //             Vec3::ZERO
    //     //         ),
    //     //         ..default()
    //     //     },
    //     //     Shape,
    //     // ));
    //
    //     let r = commands.spawn((
    //         PbrBundle {
    //             mesh: meshes.add(Mesh::from(shape::Box::from_corners(
    //                               Vec3::new(-1., -1., 0.,),
    //                               Vec3::new(1., 1., 100.,),
    //                   ))),
    //             material: red,
    //             transform: Transform::from_xyz(0., 0., 0.,),
    //             ..default()
    //         },
    //         Shape,
    //     )).id();
    //     let g = commands.spawn((
    //         PbrBundle {
    //             mesh: meshes.add(Mesh::from(shape::Box::from_corners(
    //                               Vec3::new(-1., 0., -1.,),
    //                               Vec3::new(1., 100., 1.,),
    //                   ))),
    //             material: green,
    //             transform: Transform::from_xyz(0., 0., 0.,),
    //             ..default()
    //         },
    //         Shape,
    //     )).id();
    //     let b = commands.spawn((
    //         PbrBundle {
    //             mesh: meshes.add(Mesh::from(shape::Box::from_corners(
    //                               Vec3::new(0., -1., -1.,),
    //                               Vec3::new(100., 1., 1.,),
    //                   ))),
    //             material: blue,
    //             transform: Transform::from_xyz(0., 0., 0.,),
    //             ..default()
    //         },
    //         Shape,
    //     )).id();
    //
    // commands.entity(parent).push_children(&[r, g, b]);
    // }

    // commands.insert_resource(CurrentImage(img));

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, -(max_distance + 1.0))
                .looking_at(Vec3::ZERO, -Vec3::Y),
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

    *bg_size = background::BackgroundSize(width as f32 * 2., height as f32 * 2.);
}

fn win_check(parent_transform: Query<&Transform, With<MyParent>>, mut bg_brightness: ResMut<background::BackgroundBrightness>, mut win_timer: ResMut<WinTimer>, time: Res<Time>) {
    for transform in &parent_transform {
        let (rot_vec, _f) = transform.rotation.to_axis_angle();
        let angle = rot_vec.normalize().cross(Vec3::X).z;
        println!("angle = {}", angle);
        if angle.abs() < 0.02 {
                println!("Angle correct. Hold to win");
            bg_brightness.0 = bg_brightness.0.map(|b| b - 0.5);
            win_timer.0 += time.delta_seconds_f64();
            if win_timer.0 > 5. {
                println!("You Win!!");
            }
        } else if win_timer.0 > 0. {
            bg_brightness.0 = None;
            win_timer.0 = 0.;
        }

    }
}

fn rotate(mut query: Query<&mut Transform, With<MyCamera>>, time: Res<Time>) {
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

fn mouse_input(
    mouse_button_input: Res<Input<MouseButton>>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut query: Query<&mut Transform, With<MyParent>>,
    time: Res<Time>,
) {
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



#[derive(Component, Deref)]
struct InstanceMaterialData(Vec<InstanceData>);

impl ExtractComponent for InstanceMaterialData {
    type Query = &'static InstanceMaterialData;
    type Filter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(InstanceMaterialData(item.0.clone()))
    }
}

pub struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<InstanceMaterialData>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSet::Queue),
                    prepare_instance_buffers.in_set(RenderSet::Prepare),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).init_resource::<CustomPipeline>();
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<Mesh>>,
    material_meshes: Query<(Entity, &MeshUniform, &Handle<Mesh>), With<InstanceMaterialData>>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<Transparent3d>)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

    for (view, mut transparent_phase) in &mut views {
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for (entity, mesh_uniform, mesh_handle) in &material_meshes {
            if let Some(mesh) = meshes.get(mesh_handle) {
                let key =
                    view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                let pipeline = pipelines
                    .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                    .unwrap();
                transparent_phase.add(Transparent3d {
                    entity,
                    pipeline,
                    draw_function: draw_custom,
                    distance: rangefinder.distance(&mesh_uniform.transform),
                });
            }
        }
    }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &InstanceMaterialData)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instance data buffer"),
            contents: bytemuck::cast_slice(instance_data.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

#[derive(Resource)]
pub struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for CustomPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/instancing.wgsl");

        let mesh_pipeline = world.resource::<MeshPipeline>();

        CustomPipeline {
            shader,
            mesh_pipeline: mesh_pipeline.clone(),
        }
    }
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        // meshes typically live in bind group 2. because we are using bindgroup 1
        // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
        // linked in the shader
        descriptor
            .vertex
            .shader_defs
            .push("MESH_BINDGROUP_1".into());

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawMeshInstanced,
);

pub struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = SRes<RenderAssets<Mesh>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = (Read<Handle<Mesh>>, Read<InstanceBuffer>);

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        (mesh_handle, instance_buffer): (&'w Handle<Mesh>, &'w InstanceBuffer),
        meshes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_mesh = match meshes.into_inner().get(mesh_handle) {
            Some(gpu_mesh) => gpu_mesh,
            None => return RenderCommandResult::Failure,
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..gpu_mesh.vertex_count, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}
