
var render_pipe;
var render_bind_group;

function setup_render_particles(uniformArray, computeStorageBuffer) {

  const cellShaderModule = device.createShaderModule({
    label: 'RenderParticle',
    code: `
       const kTri = array(
      //   X,    Y,
      -1.0, -1.0, // Triangle 1 (Blue)
       1.0, -1.0,
       1.0,  1.0,

      -1.0, -1.0, // Triangle 2 (Red)
       1.0,  1.0,
      -1.0,  1.0);

        struct VertexInput {
          @builtin(instance_index) idx: u32,
        };

        struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) vert_pos: vec2f, // Not used yet
        };
        
        struct FragInput {
          @location(0) vert_pos: vec2f,
        };

        struct Particle {
           pos: vec2i,
           vel: vec2f,
           id: vec2u,
        };
        

        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage> cellStateOut: array<Particle>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var posConst = array<vec2f, 3>(
          vec2(0.0, 0.5),
          vec2(-0.5, -0.5),
          vec2(0.5, -0.5)
        );

          var output: VertexOutput;
          output.pos = vec4f(posConst[input.idx], 0.01, 1);
          output.vert_pos = vec2f(0,0);
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {

          return vec4f(1,1,1 ,0.5) ;
        }
      `
  });

  
      // Create the bind group layout and pipeline layout.
      const bindGroupLayout = device.createBindGroupLayout({
        label: "Cell Bind Group Layout",
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE |  GPUShaderStage.FRAGMENT,
          buffer: {} // Grid uniform buffer
        }, {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE |  GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage"} // Cell state input buffer
        }]
      });
      
        
    const pipelineLayout = device.createPipelineLayout({
      label: "Cell Pipeline Layout",
      bindGroupLayouts: [bindGroupLayout],
    });


    render_pipe = device.createRenderPipeline({
    label: "Cell pipeline",
    primitive: {cullMode:"none"},
    layout: pipelineLayout, // Updated!
    vertex: {
      module: cellShaderModule,
      entryPoint: "vertexMain",
      buffers: []
    },
    fragment: {
      module: cellShaderModule,
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


   render_bind_group = device.createBindGroup({
    label: "Compute renderer bind group A",
    layout: bindGroupLayout, // Updated Line
    entries: [{
      binding: 0,
      resource: { buffer: uniformArray }
    }, {
      binding: 1, // New Entry
      resource: { buffer: computeStorageBuffer },
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
  pass.setPipeline(render_pipe);

  pass.setBindGroup(0, render_bind_group); // Updated!
  pass.draw(3);

  // End the render pass and submit the command buffer
  pass.end();
}


