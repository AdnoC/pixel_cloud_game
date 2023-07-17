//! This example demonstrates the built-in 3d shapes in Bevy.
//! The scene includes a patterned texture and a rotation for visualizing the normals and UVs.

use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use std::{
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
use image::{io::Reader as ImageReader, Rgba, RgbaImage};
use palette::{IntoColor, Srgb};

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
struct ImageName(String);

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
            CustomMaterialPlugin,
        ))
        .add_systems(Update, init_cloud)
        .insert_resource(ImageName("goomba.png".to_string()))
        // .add_systems(Startup, setup)
        .add_systems(Update, (load_dd_level, mouse_grab, mouse_input, rotate))
        .run();
}

fn load_dd_level(mut img_name: ResMut<ImageName>, mut dnd_evr: EventReader<FileDragAndDrop>) {
    for ev in dnd_evr.iter() {
       if let FileDragAndDrop::DroppedFile { path_buf, .. } = ev {
            println!("Dropped file with path: {:?}", path_buf);
        if let Some(filename) = path_buf.to_str() {

            *img_name = ImageName(filename.to_string());
        }
        }
    }
}

fn init_cloud(
    mut commands: Commands,
    img_name: Res<ImageName>,
    mut meshes: ResMut<Assets<Mesh>>,
    preexisting: Query<Entity, Or<(With<MyParent>, With<MyCamera>)>>
) {
    if !img_name.is_changed() {
        return;
    }
    for entity in preexisting.iter() {
        commands.entity(entity).despawn_recursive();
    }

    let img = ImageReader::open(&img_name.0)
    // let img = ImageReader::open("goomba.png")
        .expect("could not find image")
        .decode()
        .expect("could not decode image")
        .to_rgba8();
    let height = img.height();
    let width = img.width();
    const RADIUS: f32 = 0.55;
    let max_distance = u32::max(height, width) as f32;

    let parent = commands.spawn((MyParent, SpatialBundle::default())).id();
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
