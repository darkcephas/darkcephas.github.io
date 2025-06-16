
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
          @location(0) uv_pos: vec2f, // Not used yet
        };
        
        struct FragInput {
          @location(0) uv_pos: vec2f,
        };

        struct Particle {
           posi: vec2i,
           id: vec2u,
           posf: vec2f,
           vel: vec2f,
        };

        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage> cellBuffer: array<Particle>;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
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
          var pos =  kTriDef[input.vdx % kVertsPerQuad] * 0.01 + cellBuffer[input.vdx/ kVertsPerQuad].posf;
          output.pos = vec4f(pos, 0, 1);
          output.uv_pos = kTriDef[input.vdx % kVertsPerQuad];
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
          let sphereAlpha = (1.0-length(input.uv_pos));
          return vec4f(sphereAlpha, sphereAlpha, sphereAlpha ,1.0);
        }
      `
  });


  // Create the bind group layout and pipeline layout.
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
      buffer: {} // Grid uniform buffer
    }, {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" } // Cell state input buffer
    }]
  });


  const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  massRenderPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout, // Updated!
    vertex: {
      module: renderShaderModule,
      entryPoint: "vertexMain",
      buffers: []
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "fragmentMain",
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
    layout: bindGroupLayout, // Updated Line
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1, // New Entry
      resource: { buffer: cellStateStorage },
    }],
  });

}

function draw_particles(encoder, step) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0.0, b: 0.2, a: 1.0 },
      storeOp: "store",
    }]
  });

  // Draw the grid.
  pass.setPipeline(massRenderPipeline);
  pass.setBindGroup(0, render_binding); // Updated!

  pass.draw(NUM_PARTICLES_PER_MICRO* NUM_MICRO_SIMS*2);
  // End the render pass and submit the command buffer
  pass.end();
}


