var device;
var canvasformat;
var context;
const NUM_MICRO_SIMS = 1;
const NUM_PARTICLES_PER_MICRO = 3; // 3 body
var canvas_width;
var canvas_height;
var bindGroupLayout;
var uniformBuffer;
var simulationBindGroups;
var massAssignBindGroups;
var starGraphicsBindGroup;
var massGraphicsBindGroup;
var forceIndexBindGroups;

function UpdateUniforms() {
  // Create a uniform buffer that describes the grid.
  const uniformArray = new Float32Array([canvas_width, canvas_height]);
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
}
window.onload = async function () {

  window.addEventListener('resize', resizeCanvas, false);
  const canvas = document.querySelector("canvas");
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    /**
     * Your drawings need to be inside this function otherwise they will be reset when 
     * you resize the browser window and the canvas goes will be cleared.
     */
    // drawStuff(); 
  }

  resizeCanvas();


  if (!canvas) {
    throw new Error("No canvas.");
  }
  canvas_width = canvas.width;
  canvas_height = canvas.height;
  // Your WebGPU code will begin here!
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  const numElementsCell = 8;
  const cellStateArray = new Float32Array(NUM_PARTICLES_PER_MICRO * NUM_MICRO_SIMS * numElementsCell);
  var as_int = new Int32Array(cellStateArray.buffer);
  var cellStateStorage =
    device.createBuffer({
      label: "Cell State A",
      size: cellStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

  for (let i = 0; i < cellStateArray.length; i += numElementsCell * NUM_PARTICLES_PER_MICRO) {
    for (let j = 0; j < NUM_PARTICLES_PER_MICRO; j++) {
      let q = i + j * numElementsCell;
      cellStateArray[q + 0] = Math.random() - 0.5;
      cellStateArray[q + 1] = Math.random() - 0.5;
      cellStateArray[q + 2] = Math.random() - 0.5;
      cellStateArray[q + 3] = Math.random() - 0.5;
      cellStateArray[q + 4] = Math.random() - 0.5;
      cellStateArray[q + 5] = Math.random() - 0.5;
      cellStateArray[q + 6] = Math.random() - 0.5;
      cellStateArray[q + 7] = Math.random() - 0.5;

      as_int[q + 0] = 0;
      as_int[q + 1] = 0;
      as_int[q + 2] = 0;
      as_int[q + 3] = 0;
    }
  }
  device.queue.writeBuffer(cellStateStorage, 0, cellStateArray);


  uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  UpdateUniforms();


  setup_compute_particles(uniformBuffer, cellStateStorage);
  setup_render_particles(uniformBuffer, cellStateStorage);

  let step = 0; // Track how many simulation steps have been run        
  function updateGrid() {
    step++; // Increment the step count

    // Start a render pass 
    const encoder = device.createCommandEncoder();
    update_compute_particles(encoder, step);
    draw_particles(encoder, step);
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
    window.requestAnimationFrame(updateGrid);
  }
  window.requestAnimationFrame(updateGrid);
}
