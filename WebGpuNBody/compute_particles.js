var simulationShaderModule
const WORKGROUP_SIZE = 8;
var simulationPipeline;
var cellStateStorage;

function setup_compute_particles(pipelineLayout) {
    

    simulationShaderModule = device.createShaderModule({
      label: "Life simulation shader",
      code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        fn cellIndex(cell: vec2u) -> u32 {
          return (cell.y % u32(grid.y)) * u32(grid.x) +
                  (cell.x % u32(grid.x));
        }

        fn cellActive(x: u32, y: u32) -> u32 {
          return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
          // Determine how many active neighbors this cell has.
          let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                                cellActive(cell.x+1, cell.y) +
                                cellActive(cell.x+1, cell.y-1) +
                                cellActive(cell.x, cell.y-1) +
                                cellActive(cell.x-1, cell.y-1) +
                                cellActive(cell.x-1, cell.y) +
                                cellActive(cell.x-1, cell.y+1) +
                                cellActive(cell.x, cell.y+1);

          let i = cellIndex(cell.xy);

          // Conway's game of life rules:
          switch activeNeighbors {
            case 2: {
              cellStateOut[i] = cellStateIn[i];
            }
            case 3: {
              cellStateOut[i] = 1;
            }
            default: {
              cellStateOut[i] = 0;
            }
          }
        }
      `
    });  

    // Create an array representing the active state of each cell.
    const cellStateArray = new Float32Array(GRID_SIZE * GRID_SIZE * 2);

    // Create two storage buffers to hold the cell state.
    cellStateStorage = [
      device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Cell State B",
         size: cellStateArray.byteLength,
         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
    ];
    // Mark every third cell of the first grid as active.
    for (let i = 0; i < cellStateArray.length; i++) {
      cellStateArray[i] =  Math.random()-0.5 ;
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);
    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);
    // Create a bind group to pass the grid uniforms into the pipeline
    
     // Create a compute pipeline that updates the game state.
    simulationPipeline = device.createComputePipeline({
      label: "Simulation pipeline",
      layout: pipelineLayout,
      compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
      }
    });
}

function update_compute_particles(encoder,bindGroups, step)
{
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(simulationPipeline);
  computePass.setBindGroup(0, bindGroups[step % 2]);
  const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
  computePass.end();
}
    
