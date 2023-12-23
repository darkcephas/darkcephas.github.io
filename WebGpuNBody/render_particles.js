
var cellPipeline;
var vertexBuffer;
var vertices;

function setup_render_particles(pipelineLayout) {
    vertices = new Float32Array([
    //   X,    Y,
      -0.8, -0.8, // Triangle 1 (Blue)
       0.8, -0.8,
       0.8,  0.8,

      -0.8, -0.8, // Triangle 2 (Red)
       0.8,  0.8,
      -0.8,  0.8,
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
          @location(0) cell: vec2f, // New line!
        };
        
        struct FragInput {
          @location(0) cell: vec2f,
        };

        @group(0) @binding(1) var<storage> cellState: array<u32>; // New!
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          let i = f32(input.instance);
          let cell = vec2f(i % grid.x, floor(i / grid.x));
          let cellOffset = cell / grid * 2;
          let state = f32(cellState[input.instance]); // New line!
          let gridPos = (input.pos*state+1) / grid - 1 + cellOffset;
     
          var output: VertexOutput;
          output.pos = vec4f(gridPos, 0, 1);
          output.cell = cell; // New line!
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
            let c = input.cell / grid;
            return vec4f(c, 1-c.x, 1);
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
          format: canvasFormat
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
          clearValue: { r: 0, g: 0.3, b: 0.1, a: 1.0 },
          storeOp: "store",
        }]
      });

      // Draw the grid.
      pass.setPipeline(cellPipeline);
      pass.setBindGroup(0, bindGroups[step % 2]); // Updated!
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);
      // End the render pass and submit the command buffer
      pass.end();
}


