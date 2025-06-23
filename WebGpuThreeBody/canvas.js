var device;
var canvasformat;
var context;
const NUM_MICRO_SIMS = 256*256*8;
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
const INT_SCALE_CANVAS = 256;

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
    canvas_width = canvas.width;
    canvas_height = canvas.height;
 

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

  var planet_pos_x = [];
  var planet_pos_y = [];
  var planet_vel_x = [];
  var planet_vel_y = [];
  var sum_vel_x = 0.0;
  var sum_vel_y = 0.0;
  for (let j = 0; j < NUM_PARTICLES_PER_MICRO; j++) {
    planet_pos_x.push( (Math.random() - 0.5)*0.5);
    planet_pos_y.push( (Math.random() - 0.5)*0.5);

    var curr_vel_x =  (Math.random() - 0.5)*3.0;
    var curr_vel_y =  (Math.random() - 0.5)*3.0;

    planet_vel_x.push( j==NUM_PARTICLES_PER_MICRO-1? -sum_vel_x: curr_vel_x);
    planet_vel_y.push( j==NUM_PARTICLES_PER_MICRO-1? -sum_vel_y: curr_vel_y);

    sum_vel_x += curr_vel_x;
    sum_vel_y += curr_vel_y;
  }

  for (let i = 0; i < cellStateArray.length; i += numElementsCell * NUM_PARTICLES_PER_MICRO) {
    for (let j = 0; j < NUM_PARTICLES_PER_MICRO; j++) {
      let q = i + j * numElementsCell;

      var curr_pos_x = planet_pos_x[j]+(Math.random() - 0.5)*0.0005;
      var curr_pos_y = planet_pos_y[j]+(Math.random() - 0.5)*0.0005;
      curr_pos_x = INT_SCALE_CANVAS * curr_pos_x;
      curr_pos_y = INT_SCALE_CANVAS * curr_pos_y;

      as_int[q + 0] = Math.floor(curr_pos_x);
      as_int[q + 1] = Math.floor(curr_pos_y);
      cellStateArray[q + 4] = 0.0;
      cellStateArray[q + 5] = 0.0;
      cellStateArray[q + 4] =  curr_pos_x - Math.floor(curr_pos_x);;
      cellStateArray[q + 5] = curr_pos_y- Math.floor(curr_pos_y);
      cellStateArray[q + 6] = planet_vel_x[j];
      cellStateArray[q + 7] = planet_vel_y[j];
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
    UpdateUniforms();
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
