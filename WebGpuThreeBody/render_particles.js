
var cellPipeline;
var vertexBuffer;
var vertices;
var massRenderPipeline;
var render_binding;

function setup_render_particles(uniformBuffer, cellStateStorage) {
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
        @group(0) @binding(1) var<storage> renderBufferIn: array<u32>;
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

          let star_count = f32(renderBufferIn[x_pixel + u32(canvas_size.x)* y_pixel])/256.0;
   
          return vec4f(1,1,1 ,1) ;

;
        }
      `
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
          return vec4f(f32(misaligned_amount)/1.0, 0.0, 0.0 ,1.0);

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
      pass.setPipeline(massRenderPipeline);
      pass.setBindGroup(0, render_binding); // Updated!
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(vertices.length / 2);
      // End the render pass and submit the command buffer
      pass.end();
}


