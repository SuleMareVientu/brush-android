#![cfg(target_os = "android")]

use jni::EnvUnowned;
use jni::objects::{JClass, JString, JByteArray};
use jni::sys::jint;
use serde::Deserialize;
use std::os::fd::FromRawFd;
use std::fs::File;
use std::io::Read;
use std::thread;

use brush_render::AlphaMode;

/// Scalable configuration matching the Desktop CLI arguments
#[derive(Deserialize, Debug)]
pub struct TrainingConfig {
    // Training options
    pub total_train_iters: u32,
    pub lr_mean: f64,
    pub lr_mean_end: f64,
    pub mean_noise_weight: f32,
    pub lr_coeffs_dc: f64,
    pub lr_coeffs_sh_scale: f32,
    pub lr_opac: f64,
    pub lr_scale: f64,
    pub lr_rotation: f64,
    pub ssim_weight: f32,
    pub opac_decay: f32,
    pub background_color: Vec<f32>,
    pub background_noise_strength: f32,
    pub random_init_scene_scale: Option<f32>,

    // Refine options
    pub max_splats: u32,
    pub refine_every: u32,
    pub growth_grad_threshold: f32,
    pub growth_select_fraction: f32,
    pub growth_stop_iter: u32,
    pub split_at_screen_size: f32,
    pub match_alpha_weight: f32,
    pub lpips_loss_weight: f32,

    // LOD options
    pub lod_levels: u32,
    pub lod_refine_steps: u32,
    pub lod_decimation_keep: u32,
    pub lod_image_scale: u32,

    // Model options
    pub sh_degree: u32,

    // Dataset options
    pub max_frames: Option<usize>,
    pub max_resolution: u32,
    pub eval_split_every: Option<usize>,
    pub subsample_frames: Option<u32>,
    pub subsample_points: Option<u32>,
    pub alpha_mode: Option<AlphaMode>,
    pub max_scene_batch_cache_size: String,

    // Process options
    pub seed: u64,
    pub start_iter: u32,
    pub eval_every: u32,
    pub eval_save_to_disk: bool,
    pub export_every: u32,
    pub export_path: String,
    pub export_name: String,
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_splats_brush_BrushEngine_startNative<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    config_json_jstr: JString<'local>,
    fd: jint,
) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    // 1. Safely accept Android File Descriptor and wrap it immediately
    if fd < 0 {
        log::error!("JNI: Invalid file descriptor provided: {}", fd);
        return;
    }

    // SAFETY: Java detached the FD, transferring ownership to Rust.
    // Wrap it immediately so early error returns safely drop the File and close the descriptor.
    let mut file = unsafe { File::from_raw_fd(fd) };

    // 2. Upgrade EnvUnowned to a JNIEnv reference using with_env to query Java String and prevent unwinding panics.
    // If an error occurs, it throws a java.lang.RuntimeException in the JVM and returns a default value ("" empty string).
    let config_json_str = env.with_env(|env| -> jni::errors::Result<String> {
        config_json_jstr.try_to_string(env)
    }).resolve::<jni::errors::ThrowRuntimeExAndDefault>();

    if config_json_str.is_empty() {
        log::error!("JNI: Received empty or invalid configuration string");
        return;
    }

    // 3. Unified Configuration Payload Ingestion (Scalable JSON parsing)
    let config: TrainingConfig = match serde_json::from_str(&config_json_str) {
        Ok(cfg) => cfg,
        Err(e) => {
            log::error!("JNI: Failed to parse configuration JSON: {:?}", e);
            return;
        }
    };
    log::info!("JNI: Loaded Configuration: {:?}", config);


    // 4. Delegate to training engine in a background thread to prevent Android UI ANRs.
    thread::spawn(move || {
        log::info!("Background thread started for native processing. Ingesting FD...");
        let mut buffer = Vec::new();
        if let Err(e) = file.read_to_end(&mut buffer) {
            log::error!("JNI: Failed to read from FD: {:?}", e);
            return;
        }
        log::info!("JNI: Successfully read {} bytes from FD", buffer.len());

        run_training_engine(config, buffer);
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_splats_brush_BrushEngine_startFromBufferNative<'local>(
    mut env: EnvUnowned<'local>,
    _class: JClass<'local>,
    config_json_jstr: JString<'local>,
    zip_bytes_jarray: JByteArray<'local>,
) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    // 1. Upgrade EnvUnowned to a JNIEnv reference
    let config_json_str = env.with_env(|env| -> jni::errors::Result<String> {
        config_json_jstr.try_to_string(env)
    }).resolve::<jni::errors::ThrowRuntimeExAndDefault>();

    if config_json_str.is_empty() {
        log::error!("JNI: Received empty or invalid configuration string");
        return;
    }

    // 2. Unified Configuration Payload Ingestion (Scalable JSON parsing)
    let config: TrainingConfig = match serde_json::from_str(&config_json_str) {
        Ok(cfg) => cfg,
        Err(e) => {
            log::error!("JNI: Failed to parse configuration JSON: {:?}", e);
            return;
        }
    };
    log::info!("JNI: Loaded Configuration: {:?}", config);

    // 3. Convert the Java byte array directly into a Rust Vec<u8>
    let buffer = env.with_env(|env| {
        env.convert_byte_array(&zip_bytes_jarray)
    }).resolve::<jni::errors::ThrowRuntimeExAndDefault>();

    if buffer.is_empty() {
        log::error!("JNI: Received empty or invalid byte array buffer");
        return;
    }
    log::info!("JNI: Successfully converted {} bytes from Java byte array", buffer.len());

    // 4. Delegate to training engine in a background thread to prevent Android UI ANRs.
    thread::spawn(move || {
        log::info!("Background thread started for native buffer processing.");
        run_training_engine(config, buffer);
    });
}

/// Headless core training engine worker.
fn run_training_engine(config: TrainingConfig, buffer: Vec<u8>) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime");

    rt.block_on(async move {
        log::info!("Tokio async processing started.");
        
        // Build the data source using the new Buffer variant.
        let source = brush_vfs::DataSource::Buffer(buffer, "dataset.zip".to_owned());

        // Build the TrainStreamConfig.
        let mut train_stream_config = brush_process::config::TrainStreamConfig::default();
        
        // 1. Training options
        train_stream_config.train_config.total_train_iters = config.total_train_iters;
        train_stream_config.train_config.lr_mean = config.lr_mean;
        train_stream_config.train_config.lr_mean_end = config.lr_mean_end;
        train_stream_config.train_config.mean_noise_weight = config.mean_noise_weight;
        train_stream_config.train_config.lr_coeffs_dc = config.lr_coeffs_dc;
        train_stream_config.train_config.lr_coeffs_sh_scale = config.lr_coeffs_sh_scale;
        train_stream_config.train_config.lr_opac = config.lr_opac;
        train_stream_config.train_config.lr_scale = config.lr_scale;
        train_stream_config.train_config.lr_rotation = config.lr_rotation;
        train_stream_config.train_config.ssim_weight = config.ssim_weight;
        train_stream_config.train_config.opac_decay = config.opac_decay;
        train_stream_config.train_config.background_color = config.background_color.clone();
        train_stream_config.train_config.background_noise_strength = config.background_noise_strength;
        train_stream_config.train_config.random_init_scene_scale = config.random_init_scene_scale;

        // 2. Refine options
        train_stream_config.train_config.max_splats = config.max_splats;
        train_stream_config.train_config.refine_every = config.refine_every;
        train_stream_config.train_config.growth_grad_threshold = config.growth_grad_threshold;
        train_stream_config.train_config.growth_select_fraction = config.growth_select_fraction;
        train_stream_config.train_config.growth_stop_iter = config.growth_stop_iter;
        train_stream_config.train_config.split_at_screen_size = config.split_at_screen_size;
        train_stream_config.train_config.match_alpha_weight = config.match_alpha_weight;
        train_stream_config.train_config.lpips_loss_weight = config.lpips_loss_weight;

        // 3. LOD options
        train_stream_config.train_config.lod_levels = config.lod_levels;
        train_stream_config.train_config.lod_refine_steps = config.lod_refine_steps;
        train_stream_config.train_config.lod_decimation_keep = config.lod_decimation_keep;
        train_stream_config.train_config.lod_image_scale = config.lod_image_scale;

        // 4. Model options
        train_stream_config.model_config.sh_degree = config.sh_degree;

        // 5. Dataset options
        train_stream_config.load_config.max_frames = config.max_frames;
        train_stream_config.load_config.max_resolution = config.max_resolution;
        train_stream_config.load_config.eval_split_every = config.eval_split_every;
        train_stream_config.load_config.subsample_frames = config.subsample_frames;
        train_stream_config.load_config.subsample_points = config.subsample_points;
        train_stream_config.load_config.alpha_mode = config.alpha_mode;
        
        if let Ok(bytes) = parse_size::parse_size(&config.max_scene_batch_cache_size) {
            train_stream_config.load_config.max_scene_batch_cache_size = bytes;
        } else {
            log::warn!("JNI: Failed to parse cache size: {}, falling back to default", config.max_scene_batch_cache_size);
        }

        // 6. Process options
        train_stream_config.process_config.seed = config.seed;
        train_stream_config.process_config.start_iter = config.start_iter;
        train_stream_config.process_config.eval_every = config.eval_every;
        train_stream_config.process_config.eval_save_to_disk = config.eval_save_to_disk;
        train_stream_config.process_config.export_every = config.export_every;
        train_stream_config.process_config.export_name = config.export_name.clone();

        if !config.export_path.is_empty() {
            let mut path = config.export_path.clone();
            if !path.ends_with('/') && !path.ends_with('\\') {
                path.push('/');
            }
            train_stream_config.process_config.export_path = path;
        }

        // Initialize the burn backend (WGPU Setup)
        brush_process::burn_init_setup().await;

        // Create the process
        let process = brush_process::create_process(source, async move |_init| {
            Some(train_stream_config)
        });

        // Pump the stream to drive training and exporting.
        use tokio_stream::StreamExt;
        let mut stream = process.stream;
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(msg) => match msg {
                    brush_process::message::ProcessMessage::NewProcess => {
                        log::info!("Android training process started");
                    }
                    brush_process::message::ProcessMessage::StartLoading { name, training, .. } => {
                        log::info!("Loading dataset {} (training: {})", name, training);
                    }
                    brush_process::message::ProcessMessage::SplatsUpdated { num_splats, frame, total_frames, .. } => {
                        log::info!("Splats updated: {} splats (frame {}/{})", num_splats, frame, total_frames);
                    }
                    brush_process::message::ProcessMessage::TrainMessage(train) => match train {
                        brush_process::message::TrainMessage::TrainStep { iter, total_elapsed, .. } => {
                            if iter % 100 == 0 || iter == 1 {
                                let mut mem_info = String::new();
                                if let Some(usage) = memory_stats::memory_stats() {
                                    mem_info = format!(
                                        " | Mem: Physical={}MB, Virtual={}MB",
                                        usage.physical_mem / (1024 * 1024),
                                        usage.virtual_mem / (1024 * 1024)
                                    );
                                }
                                log::info!(
                                    "Training step {}/{} - Elapsed: {:?}{}",
                                    iter,
                                    config.total_train_iters,
                                    total_elapsed,
                                    mem_info
                                );
                            }
                        }
                        brush_process::message::TrainMessage::RefineStep { cur_splat_count, iter, .. } => {
                            log::info!("Refine step at iter {}: {} splats", iter, cur_splat_count);
                        }
                        brush_process::message::TrainMessage::EvalResult { iter, avg_psnr, avg_ssim } => {
                            log::info!("Eval at iter {}: PSNR = {}, SSIM = {}", iter, avg_psnr, avg_ssim);
                        }
                        brush_process::message::TrainMessage::Dataset { dataset } => {
                            log::info!("Dataset loaded. Train views: {}, Eval views: {:?}", 
                                dataset.train.views.len(), 
                                dataset.eval.as_ref().map_or(0, |e| e.views.len())
                            );
                        }
                        brush_process::message::TrainMessage::DoneTraining => {
                            log::info!("Training complete!");
                        }
                        _ => {}
                    }
                    brush_process::message::ProcessMessage::DoneLoading => {
                        log::info!("Finished loading dataset.");
                    }
                    brush_process::message::ProcessMessage::Warning { error } => {
                        log::warn!("Process Warning: {}", error);
                    }
                }
                Err(e) => {
                    log::error!("Training stream error: {:?}", e);
                    break;
                }
            }
        }

        log::info!("Tokio async processing complete.");
    });
}
