
var cellPipeline;
var vertexBuffer;
var vertices;
var massRenderPipeline;
var render_binding;

function setup_render_particles(uniformBuffer, cellStateStorage) {



  const massRenderShaderModule = device.createShaderModule({
    label: 'Cell shader',
    code: `
        struct VertexInput {

          @builtin(vertex_index) vdx: u32,
          @builtin(instance_index) idx: u32,
        };

        struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) vert_pos: vec2f, // Not used yet
        };
        
        struct FragInput {
          @location(0) vert_pos: vec2f,
        };


        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage> renderBufferIn: array<vec4u>;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var pos = array<vec2f, 3>(
            vec2(  -1.0, -1.0),
            vec2(1.0, -1.0),
            vec2( 1.0, 1.0)
          );
          var output: VertexOutput;

          output.pos = vec4f(pos[input.vdx], 0, 1);
          output.vert_pos = pos[input.idx]*0.1;
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
          return vec4f(1, 0.0, 0.3 ,1.0);

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
      module: massRenderShaderModule,
      entryPoint: "vertexMain",
      buffers: []
    },
    fragment: {
      module: massRenderShaderModule,
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

  pass.draw(3);
  // End the render pass and submit the command buffer
  pass.end();
}


