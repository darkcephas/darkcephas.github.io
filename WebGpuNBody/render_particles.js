
var cellPipeline;
var vertexBuffer;
var vertices;

function setup_render_particles(pipelineLayout) {
    vertices = new Float32Array([
    //   X,    Y,
      -1.0, -1.0, // Triangle 1 (Blue)
       1.0, -1.0,
       1.0,  1.0,

      -1.0, -1.0, // Triangle 2 (Red)
       1.0,  1.0,
      -1.0,  1.0,
    ]);
    vertexBuffer = device.createBuffer({
      label: "Cell vertices",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);
    
    const vertexBufferLayout = {
      arrayStride: 8,
      attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0, // Position, see vertex shader
      }],
    };
    
    const cellShaderModule = device.createShaderModule({
      label: 'Cell shader',
      code: `
        struct VertexInput {
          @location(0) pos: vec2f,
          @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
          @builtin(position) pos: vec4f,
          @location(0) vert_pos: vec2f, // Not used yet
        };
        
        struct FragInput {
          @location(0) vert_pos: vec2f,
        };

        struct Particle {
          pos: vec2f,
          vel: vec2f,
       };


        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<Particle>;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          let i = f32(input.instance);
          let gridPos = (input.pos.xy / grid) +  cellState[input.instance].pos;
          output.pos = vec4f(gridPos, 0, 1);
          output.vert_pos = input.pos.xy; // New line!
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
            var r_res = 1.0-dot(input.vert_pos,input.vert_pos);
            r_res = max(r_res, 0.0);
            return vec4f(r_res*0.1,r_res*0.2,r_res*0.3,1);
        }
      `
    });
    
    cellPipeline = device.createRenderPipeline({
      label: "Cell pipeline",
      layout: pipelineLayout, // Updated!
      vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout]
      },
      fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
          format: canvasFormat,
          blend: {
            alpha: {
              dstFactor: "zero",
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
}

function draw_particles(encoder, bindGroups, step)
{
       const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: "store",
        }]
      });

      // Draw the grid.
      pass.setPipeline(cellPipeline);
      pass.setBindGroup(0, bindGroups[step % 2]); // Updated!
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(vertices.length / 2, NUM_PARTICLES_DIM * NUM_PARTICLES_DIM);
      // End the render pass and submit the command buffer
      pass.end();
}


