package com.splats.brush

import android.content.Context
import android.net.Uri
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Ergonomic Data Class mapping precisely to the Rust CLI arguments
 */
data class BrushConfig(
    // Training options
    var totalTrainIters: Int = 30000,
    var lrMean: Double = 2e-5,
    var lrMeanEnd: Double = 2e-7,
    var meanNoiseWeight: Float = 50.0f,
    var lrCoeffsDc: Double = 2e-3,
    var lrCoeffsShScale: Float = 10.0f,
    var lrOpac: Double = 0.012,
    var lrScale: Double = 5e-3,
    var lrRotation: Double = 2e-3,
    var ssimWeight: Float = 0.2f,
    var opacDecay: Float = 0.004f,
    var backgroundColor: FloatArray = floatArrayOf(0.0f, 0.0f, 0.0f),
    var backgroundNoiseStrength: Float = 0.1f,
    var randomInitSceneScale: Float? = null,

    // Refine options
    var maxSplats: Int = 10000000,
    var refineEvery: Int = 200,
    var growthGradThreshold: Float = 0.0025f,
    var growthSelectFraction: Float = 0.25f,
    var growthStopIter: Int = 15000,
    var splitAtScreenSize: Float = 0.5f,
    var matchAlphaWeight: Float = 0.1f,
    var lpipsLossWeight: Float = 0.0f,

    // LOD options
    var lodLevels: Int = 0,
    var lodRefineSteps: Int = 5000,
    var lodDecimationKeep: Int = 50,
    var lodImageScale: Int = 50,

    // Model options
    var shDegree: Int = 3,

    // Dataset options
    var maxFrames: Int? = null,
    var maxResolution: Int = 1920,
    var evalSplitEvery: Int? = null,
    var subsampleFrames: Int? = null,
    var subsamplePoints: Int? = null,
    var alphaMode: String? = null, // "masked" or "transparent"
    var maxSceneBatchCacheSize: String = "6GiB",

    // Process options
    var seed: Long = 42,
    var startIter: Int = 0,
    var evalEvery: Int = 1000,
    var evalSaveToDisk: Boolean = false,
    var exportEvery: Int = 5000,
    var exportPath: String = "",
    var exportName: String = "export_{iter}.ply"
) {
    // Zero-dependency JSON mapping utilizing Android's built-in org.json
    internal fun toJson(context: Context): String {
        val json = JSONObject()
        
        // Training options
        json.put("total_train_iters", totalTrainIters)
        json.put("lr_mean", lrMean)
        json.put("lr_mean_end", lrMeanEnd)
        json.put("mean_noise_weight", meanNoiseWeight)
        json.put("lr_coeffs_dc", lrCoeffsDc)
        json.put("lr_coeffs_sh_scale", lrCoeffsShScale)
        json.put("lr_opac", lrOpac)
        json.put("lr_scale", lrScale)
        json.put("lr_rotation", lrRotation)
        json.put("ssim_weight", ssimWeight)
        json.put("opac_decay", opacDecay)
        
        val bgArray = JSONArray()
        for (color in backgroundColor) {
            bgArray.put(color)
        }
        json.put("background_color", bgArray)
        json.put("background_noise_strength", backgroundNoiseStrength)
        randomInitSceneScale?.let { json.put("random_init_scene_scale", it) }

        // Refine options
        json.put("max_splats", maxSplats)
        json.put("refine_every", refineEvery)
        json.put("growth_grad_threshold", growthGradThreshold)
        json.put("growth_select_fraction", growthSelectFraction)
        json.put("growth_stop_iter", growthStopIter)
        json.put("split_at_screen_size", splitAtScreenSize)
        json.put("match_alpha_weight", matchAlphaWeight)
        json.put("lpips_loss_weight", lpipsLossWeight)

        // LOD options
        json.put("lod_levels", lodLevels)
        json.put("lod_refine_steps", lodRefineSteps)
        json.put("lod_decimation_keep", lodDecimationKeep)
        json.put("lod_image_scale", lodImageScale)

        // Model options
        json.put("sh_degree", shDegree)

        // Dataset options
        maxFrames?.let { json.put("max_frames", it) }
        json.put("max_resolution", maxResolution)
        evalSplitEvery?.let { json.put("eval_split_every", it) }
        subsampleFrames?.let { json.put("subsample_frames", it) }
        subsamplePoints?.let { json.put("subsample_points", it) }
        alphaMode?.let { json.put("alpha_mode", it) }
        json.put("max_scene_batch_cache_size", maxSceneBatchCacheSize)

        // Process options
        json.put("seed", seed)
        json.put("start_iter", startIter)
        json.put("eval_every", evalEvery)
        json.put("eval_save_to_disk", evalSaveToDisk)
        json.put("export_every", exportEvery)
        json.put("export_path", if (exportPath.isEmpty()) context.filesDir.absolutePath else exportPath)
        json.put("export_name", exportName)

        return json.toString()
    }
}

object BrushEngine {
    private const val TAG = "BrushEngine"

    init {
        try {
            System.loadLibrary("brush_aar")
            Log.i(TAG, "Successfully loaded native library: brush_aar")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library brush_aar", e)
        }
    }

    @JvmStatic
    private external fun startNative(configJson: String, fd: Int)

    @JvmStatic
    private external fun startFromBufferNative(configJson: String, zipBytes: ByteArray)

    /**
     * Fluently initiate native training. Safely extracts file descriptors via Android SAF
     * and couples them with the serialized configuration payload.
     */
    @JvmStatic
    fun start(context: Context, uri: Uri, config: BrushConfig = BrushConfig()) {
        try {
            context.contentResolver.openFileDescriptor(uri, "r")?.let { pfd ->
                // Ownership transfer: Detach FD out of ParcelFileDescriptor to prevent premature Java GC closure.
                val fd = pfd.detachFd()

                val jsonPayload = config.toJson(context)
                Log.i(TAG, "Dispatching configuration to native: $jsonPayload")
                
                // Dispatch config string and fd directly to Rust JNI
                startNative(jsonPayload, fd)
            } ?: Log.e(TAG, "Failed to open ParcelFileDescriptor for URI: $uri")
        } catch (e: Exception) {
            Log.e(TAG, "Exception resolving URI: $uri", e)
        }
    }

    /**
     * Fluently initiate native training directly from an in-memory ZIP byte array.
     */
    @JvmStatic
    fun start(context: Context, zipBytes: ByteArray, config: BrushConfig = BrushConfig()) {
        try {
            val jsonPayload = config.toJson(context)
            Log.i(TAG, "Dispatching configuration and memory buffer to native: $jsonPayload")
            startFromBufferNative(jsonPayload, zipBytes)
        } catch (e: Exception) {
            Log.e(TAG, "Exception starting from memory buffer", e)
        }
    }
}
