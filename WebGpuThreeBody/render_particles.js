
var cellPipeline;
var vertexBuffer;
var vertices;
var massRenderPipeline;
var render_binding;

function setup_render_particles(uniformBuffer, cellStateStorage) {

  const renderShaderModule = device.createShaderModule({
    label: 'RenderShader',
    code: `
        struct VertexInput {
          @builtin(vertex_index) vdx: u32,
          @builtin(instance_index) idx: u32,
        };

        struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) uv_pos: vec2f,
           @location(1) col: vec4f,
        };
        
        struct FragInput {
          @location(0) uv_pos: vec2f,
          @location(1) col: vec4f,
        };

        struct Particle {
           posi: vec2i,
           id: vec2f,
           posf: vec2f,
           vel: vec2f,
        };

        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage> cellBuffer: array<Particle>;
        @vertex
        fn mainvs(input: VertexInput) -> VertexOutput {
           const kVertsPerQuad = 6;
          var kTriDef = array<vec2f, kVertsPerQuad>(
            vec2(-1.0, -1.0),
            vec2(1.0, -1.0),
            vec2(1.0, 1.0),
            vec2(-1.0, -1.0),
            vec2(1.0,  1.0),
            vec2(-1.0,  1.0),
          );
          var output: VertexOutput;
          const int_scale_canvas = i32 (${INT_SCALE_CANVAS} );
          const float_scale_canvas = f32(int_scale_canvas);
          var cellPos = vec2f(cellBuffer[input.vdx/ kVertsPerQuad].posi)/float_scale_canvas + cellBuffer[input.vdx/ kVertsPerQuad].posf/float_scale_canvas ;

          var pos =  kTriDef[input.vdx % kVertsPerQuad] * 0.007 + cellPos;
          var ratio = canvas_size.x/canvas_size.y;
          output.pos = vec4f(pos.xy * vec2f(1.0f/ratio, 1.0f), 0, 1);
          var kColChange = array<vec4f, 3>(
            vec4f(1.0, 0.05,0.05,1.),
            vec4f(.05, 1.,0.05,1.),
            vec4f(0.05, 0.05,1.,1.),
          );
          output.col =  kColChange[(input.vdx/ kVertsPerQuad) % 3];
          //output.col = vec4f(1.0, cellBuffer[input.vdx/ kVertsPerQuad].id, 1.0 );

          output.uv_pos = kTriDef[input.vdx % kVertsPerQuad];
          return output;
        }
       @fragment
        fn mainfs(input: FragInput) -> @location(0) vec4f {
          let sphereAlpha = clamp(1.0-length(input.uv_pos),0.0,1.0);
          var g_mult = 0.03;
          let colOut = sphereAlpha * input.col *sphereAlpha* g_mult;
          return vec4f(colOut.rgb ,1.0);
        }
      `
  });


  const bindGroupLayout = device.createBindGroupLayout({
    label: "Render",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
      buffer: {} // uniform
    }, {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" } // Sim buffer
    }]
  });


  const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  massRenderPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
    vertex: {
      module: renderShaderModule,
      entryPoint: "mainvs",
      buffers: []
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "mainfs",
      targets: [{
        format: canvasFormat,
        blend: {
          alpha: {
            dstFactor: "one",
            srcFactor: "one",
            operation: "add"
          },
          color: {
            dstFactor: "one",
            srcFactor: "one",
            operation: "add"
          },
        },
      }]
    }
  });

  render_binding = device.createBindGroup({
    label: "Compute renderer bind group A",
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1,
      resource: { buffer: cellStateStorage },
    }],
  });

}

function draw_particles(encoder, step) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0.0, b: 0.01, a: 1.0 },
      storeOp: "store",
    }]
  });

  pass.setPipeline(massRenderPipeline);
  pass.setBindGroup(0, render_binding);
 // pass.draw(NUM_PARTICLES_PER_MICRO* NUM_MICRO_SIMS*2);
  pass.draw(NUM_MICRO_SIMS* NUM_PARTICLES_PER_MICRO*2*3)
  pass.end(); // Will be submitted by command encoder
}


