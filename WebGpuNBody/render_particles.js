
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
          @location(0) cell: vec2f, // Not used yet
        };
        
        struct FragInput {
          @location(0) cell: vec2f,
        };


        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<vec2f>;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          let i = f32(input.instance);
          let gridPos = (input.pos.xy / grid) +  cellState[input.instance];
          output.pos = vec4f(gridPos, 0, 1);
          output.cell = vec2f(0,0); // New line!
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
            return vec4f(1,1, 1, 1);
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
          clearValue: { r: 0, g: 0.0, b: 0.0, a: 1.0 },
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


