
var cellPipeline;
var vertexBuffer;
var vertices;
var massRenderPipeline;

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



        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage> renderBufferIn: array<vec4f>;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          let gridPos = input.pos.xy;
          output.pos = vec4f(gridPos, 0, 1);
          output.vert_pos = (input.pos.xy+1) *0.5* canvas_size; // New line!
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
          let x_pixel = u32(input.vert_pos.x);
          let y_pixel = u32(input.vert_pos.y);


          return renderBufferIn[x_pixel + u32(canvas_size.x)* y_pixel];

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


    const massRenderShaderModule = device.createShaderModule({
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



        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage> renderBufferIn: array<vec4u>;
        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          let gridPos = input.pos.xy;
          output.pos = vec4f(gridPos, 0, 1);
          output.vert_pos = (input.pos.xy+1) *0.5* canvas_size; // New line!
          return output;
        }
       @fragment
        fn fragmentMain(input: FragInput) -> @location(0) vec4f {
          let x_pixel = u32(input.vert_pos.x);
          let y_pixel = u32(input.vert_pos.y);


          let mass_assign_data = renderBufferIn[x_pixel/6+  (y_pixel/6)*128u];
          var misaligned_amount = 0u;
          if(mass_assign_data.x > 0)
          {
            misaligned_amount =(1+mass_assign_data.z-mass_assign_data.y)-mass_assign_data.x;
          }
          return vec4f(f32(mass_assign_data.x)/50.0, 0, 0 ,1);

        }
      `
    });
    
    massRenderPipeline = device.createRenderPipeline({
      label: "Cell pipeline",
      layout: pipelineLayout, // Updated!
      vertex: {
        module: massRenderShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout]
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
}

function draw_particles(encoder, step)
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
      pass.setBindGroup(0, starGraphicsBindGroup); // Updated!
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(vertices.length / 2);
      // End the render pass and submit the command buffer
   

      //Draw the grid.
      pass.setPipeline(massRenderPipeline);
      pass.setBindGroup(0, massGraphicsBindGroup); // Updated!
      pass.setVertexBuffer(0, vertexBuffer);
      //pass.draw(vertices.length / 2);
      // End the render pass and submit the command buffer
      pass.end();
}


