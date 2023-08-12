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
    f32::consts::{PI, TAU},
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

#[derive(Resource)]
struct ImageData(RgbaImage);
impl ImageData {
    fn new(img: RgbaImage) -> ImageData {
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
        ImageData(image::imageops::crop(&mut img, min_x, min_y, width, height).to_image())
    }
}

#[derive(Resource, Deref)]
struct WasmReceiver(Receiver<Vec<u8>>);

#[derive(Resource)]
struct WinTimer(f64);

#[cfg(target_family = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    pub fn load_image(img: &[u8]) {
        match crate::IMG_QUEUE.get() {
            Some(tx) => tx.send(img.to_vec()).expect("error sending wasm img data"),
            None => return,
        }
    }
}

static IMG_QUEUE: OnceLock<Sender<Vec<u8>>> = OnceLock::new();

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
                        ..default()
                    }),
                    ..default()
                }),
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            CustomMaterialPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, init_cloud)
        .insert_resource(ImageData(img))
        .insert_resource(WinTimer(0.0))
        // .add_systems(Startup, setup)
        .add_systems(Update, (load_external_level, load_dd_level, mouse_grab, mouse_input, rotate, win_check));

        //#[cfg(not(target_family = "wasm"))]
        {
            app.add_plugins(BackgroundPlugin);
        }
        app.run();
}


fn setup(mut commands: Commands) {
    let (tx, rx) = bounded::<Vec<u8>>(10);
    commands.insert_resource(WasmReceiver(rx));
    IMG_QUEUE.set(tx).expect("could not initialize wasm image queue");
}

fn load_external_level(mut img_data: ResMut<ImageData>, receiver: Res<WasmReceiver>) {
    for from_external in receiver.try_iter() {
        let img = load_from_memory(&from_external)
            .expect("could not find image")
            .to_rgba8();
        *img_data = ImageData::new(img);
    }
}

fn load_dd_level(mut img_data: ResMut<ImageData>, mut dnd_evr: EventReader<FileDragAndDrop>) {
    for ev in dnd_evr.iter() {
        if let FileDragAndDrop::DroppedFile { path_buf, .. } = ev {
            println!("Dropped file with path: {:?}", path_buf);
            if let Some(filename) = path_buf.to_str() {

                let img = ImageReader::open(filename)
                    .expect("could not find image")
                    .decode()
                    .expect("could not decode image")
                    .to_rgba8();
                *img_data = ImageData::new(img);
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


        //parent_bundle.transform.rotation = quat;

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

    {
        let red = materials.add(StandardMaterial {
            base_color: Color::Rgba {
                red: 1.,
                blue: 0.,
                green: 0.,
                alpha: 1.,
            },
            ..default()
        });
        let green = materials.add(StandardMaterial {
            base_color: Color::Rgba {
                red: 0.,
                blue: 0.,
                green: 1.,
                alpha: 1.,
            },
            ..default()
        });
        let blue = materials.add(StandardMaterial {
            base_color: Color::Rgba {
                red: 0.,
                blue: 1.,
                green: 0.,
                alpha: 1.,
            },
            ..default()
        });


        // commands.spawn((
        //     PbrBundle {
        //         mesh: meshes.add(Mesh::from(shape::Cylinder {
        //             radius: 5.,
        //             height: 100.,
        //             resolution: 16,
        //             segments: 2,
        //
        //         })),
        //         material: red,
        //         transform: Transform::default().looking_at(
        //             Vec3::new(1., 0., 0.,),
        //             Vec3::ZERO
        //         ),
        //         ..default()
        //     },
        //     Shape,
        // ));
        // commands.spawn((
        //     PbrBundle {
        //         mesh: meshes.add(Mesh::from(shape::Cylinder {
        //             radius: 5.,
        //             height: 100.,
        //             resolution: 16,
        //             segments: 2,
        //
        //         })),
        //         material: green,
        //         transform: Transform::default().looking_at(
        //             Vec3::new(0., 1., 0.,),
        //             Vec3::ZERO
        //         ),
        //         ..default()
        //     },
        //     Shape,
        // ));
        // commands.spawn((
        //     PbrBundle {
        //         mesh: meshes.add(Mesh::from(shape::Cylinder {
        //             radius: 5.,
        //             height: 100.,
        //             resolution: 16,
        //             segments: 2,
        //
        //         })),
        //         material: blue,
        //         transform: Transform::default().looking_at(
        //             Vec3::new(1., 0., 0.,),
        //             Vec3::ZERO
        //         ),
        //         ..default()
        //     },
        //     Shape,
        // ));

        let r = commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Box::from_corners(
                                  Vec3::new(-1., -1., 0.,),
                                  Vec3::new(1., 1., 100.,),
                      ))),
                material: red,
                transform: Transform::from_xyz(0., 0., 0.,),
                ..default()
            },
            Shape,
        )).id();
        let g = commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Box::from_corners(
                                  Vec3::new(-1., 0., -1.,),
                                  Vec3::new(1., 100., 1.,),
                      ))),
                material: green,
                transform: Transform::from_xyz(0., 0., 0.,),
                ..default()
            },
            Shape,
        )).id();
        let b = commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Box::from_corners(
                                  Vec3::new(0., -1., -1.,),
                                  Vec3::new(100., 1., 1.,),
                      ))),
                material: blue,
                transform: Transform::from_xyz(0., 0., 0.,),
                ..default()
            },
            Shape,
        )).id();

    commands.entity(parent).push_children(&[r, g, b]);
    }

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
        // let (x, y, z) = transform.rotation.to_axis_angle();//.to_euler(EulerRot::XYZ);
        let (rot_vec, _f) = transform.rotation.to_axis_angle();//.to_euler(EulerRot::XYZ);
        let rot_vec = rot_vec.normalize();
        // println!("x,y,z = {:?}", rot_vec);
        let mut dist = rot_vec.z;
        if dist > 0.5 {
            dist -= 1.;
        }
        dist = dist.abs();
        if dist.abs() < 0.001 {
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
