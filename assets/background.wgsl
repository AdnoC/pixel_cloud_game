// This shader is inspired by 70s Melt by tomorrowevening
// https://www.shadertoy.com/view/XsX3zl

#import bevy_sprite::mesh2d_vertex_output MeshVertexOutput

struct BackgroundMaterial {
    time: f32,
    width: f32,
    height: f32,
    brightness: f32,
};

@group(1) @binding(0)
var<uniform> background: BackgroundMaterial;



const zoom: i32 = 40;
//const brightness: f32 = 0.975;
var<private> fScale: f32 = 1.25;

fn cosRange(degrees: f32, range: f32, minimum: f32) -> f32 {
// RADIANS 0.017453292519943295
        return (((1.0 + cos(degrees * 0.017453292519943295)) * 0.5) * range) + minimum;
}


@fragment
fn fragment(
    in: MeshVertexOutput
) -> @location(0) vec4<f32> {
//void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
//let dir = vec3<f32>(in.uv * zoom, 1.0);
        let fragCoord = in.world_position.xy;
        let iResolution = vec2<f32>(background.width, background.height);

        let time = background.time * 1.25;
        //let time = iTime * 1.25;
        // vec2 uv = fragCoord.xy / iResolution.xy;
        var p  = (2.0*fragCoord.xy-iResolution.xy)/max(iResolution.x,iResolution.y);
        let ct = cosRange(time*5.0, 3.0, 1.1);
        let xBoost = cosRange(time*0.2, 5.0, 5.0);
        let yBoost = cosRange(time*0.1, 10.0, 5.0);

        fScale = cosRange(time * 15.5, 1.25, 0.5);

        for(var i=1;i<zoom;i = i +1) {
                let _i = f32(i);
                var newp=p;
                newp.x+=0.25/_i*sin(_i*p.y+time*cos(ct)*0.5/20.0+0.005*_i)*fScale+xBoost;
                newp.y+=0.25/_i*sin(_i*p.x+time*ct*0.3/40.0+0.03*f32(i+15))*fScale+yBoost;
                p=newp;
        }

        var col=vec3(0.5*sin(3.0*p.x)+0.5,0.5*sin(3.0*p.y)+0.5,sin(p.x+p.y));
        col *= background.brightness;

    // Add border
    let vigAmt = 5.0;
    let vignette = (1.0 - vigAmt*(in.uv.y - 0.5)*(in.uv.y - 0.5))*(1.0 - vigAmt*(in.uv.x - 0.5)*(in.uv.x - 0.5));
    var extrusion = (col.x + col.y + col.z) / 4.0;
    extrusion *= 1.5;
    extrusion *= vignette;

        // fragColor = vec4(col, extrusion);
    return vec4(col, extrusion);
}
