# VisionGuide Pro API - Android Kotlin Entegrasyon Rehberi

Bu doküman, VisionGuide Pro API'sini Android Kotlin uygulamanıza entegre etmek için gereken tüm adımları içerir.

## İçindekiler

1. [Gradle Bağımlılıkları](#gradle-bağımlılıkları)
2. [Retrofit2 API Interface](#retrofit2-api-interface)
3. [Data Modelleri](#data-modelleri)
4. [Bitmap'ten Base64 Dönüşümü](#bitmapten-base64-dönüşümü)
5. [API Servis Kullanımı](#api-servis-kullanımı)
6. [Hata Yönetimi](#hata-yönetimi)
7. [Tam Örnek Kod](#tam-örnek-kod)

---

## Gradle Bağımlılıkları

`app/build.gradle.kts` dosyanıza aşağıdaki bağımlılıkları ekleyin:

```kotlin
dependencies {
    // Retrofit2 ve OkHttp
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
    
    // Gson
    implementation("com.google.code.gson:gson:2.10.1")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
}
```

---

## Retrofit2 API Interface

`api/VisionApiService.kt` dosyasını oluşturun:

```kotlin
package com.yourpackage.api

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

interface VisionApiService {
    
    @POST("detect")
    suspend fun detectObjects(
        @Header("X-API-KEY") apiKey: String,
        @Body request: DetectionRequest
    ): Response<DetectionResponse>
}
```

---

## Data Modelleri

### DetectionRequest.kt

```kotlin
package com.yourpackage.model

import com.google.gson.annotations.SerializedName

data class DetectionRequest(
    @SerializedName("image")
    val image: String,  // Base64 encoded string
    
    @SerializedName("mode")
    val mode: String    // "E", "S", veya "M"
)
```

### DetectionResponse.kt

```kotlin
package com.yourpackage.model

import com.google.gson.annotations.SerializedName

data class DetectionResponse(
    @SerializedName("success")
    val success: Boolean,
    
    @SerializedName("mode")
    val mode: String,
    
    @SerializedName("mode_name")
    val modeName: String,
    
    @SerializedName("detections")
    val detections: List<DetectionResult>,
    
    @SerializedName("image_width")
    val imageWidth: Int,
    
    @SerializedName("image_height")
    val imageHeight: Int,
    
    @SerializedName("processing_time_ms")
    val processingTimeMs: Double
)

data class DetectionResult(
    @SerializedName("label")
    val label: String,  // Türkçe
    
    @SerializedName("label_en")
    val labelEn: String,  // İngilizce
    
    @SerializedName("score")
    val score: Double,
    
    @SerializedName("box")
    val box: BoundingBox,  // Normalized (0-1)
    
    @SerializedName("box_pixels")
    val boxPixels: BoundingBox  // Pixel cinsinden
)

data class BoundingBox(
    @SerializedName("x1")
    val x1: Double,
    
    @SerializedName("y1")
    val y1: Double,
    
    @SerializedName("x2")
    val x2: Double,
    
    @SerializedName("y2")
    val y2: Double
)
```

---

## Bitmap'ten Base64 Dönüşümü

`utils/ImageUtils.kt` dosyasını oluşturun:

```kotlin
package com.yourpackage.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import java.io.ByteArrayOutputStream

object ImageUtils {
    
    /**
     * Bitmap'i Base64 string'e çevir (NO_WRAP ile)
     * 
     * @param bitmap Dönüştürülecek bitmap
     * @param quality JPEG kalitesi (0-100, varsayılan: 85)
     * @return Base64 encoded string (NO_WRAP)
     */
    fun bitmapToBase64(bitmap: Bitmap, quality: Int = 85): String {
        val outputStream = ByteArrayOutputStream()
        
        // JPEG formatında sıkıştır
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        
        // Base64'e çevir (NO_WRAP = satır sonu karakteri ekleme)
        val imageBytes = outputStream.toByteArray()
        return Base64.encodeToString(imageBytes, Base64.NO_WRAP)
    }
    
    /**
     * Dosya yolundan Base64 string oluştur
     */
    fun imagePathToBase64(imagePath: String, quality: Int = 85): String? {
        return try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
            bitmapToBase64(bitmap, quality)
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Byte array'den Base64 string oluştur
     */
    fun byteArrayToBase64(byteArray: ByteArray): String {
        return Base64.encodeToString(byteArray, Base64.NO_WRAP)
    }
}
```

---

## API Servis Kullanımı

### Retrofit Client Oluşturma

`api/RetrofitClient.kt` dosyasını oluşturun:

```kotlin
package com.yourpackage.api

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitClient {
    
    private const val BASE_URL = "https://your-api-domain.com/"  // API base URL'inizi buraya yazın
    
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY  // Debug için, production'da NONE
    }
    
    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
    
    val apiService: VisionApiService = retrofit.create(VisionApiService::class.java)
}
```

---

## API Servis Kullanımı

### VisionRepository.kt

```kotlin
package com.yourpackage.repository

import com.yourpackage.api.RetrofitClient
import com.yourpackage.model.DetectionRequest
import com.yourpackage.model.DetectionResponse
import com.yourpackage.utils.ImageUtils
import android.graphics.Bitmap

class VisionRepository {
    
    private val apiService = RetrofitClient.apiService
    private val apiKey = "your-secret-api-key-here"  // API anahtarınızı buraya yazın
    
    /**
     * Bitmap görüntüsünden nesne tanıma yap
     * 
     * @param bitmap Tespit edilecek görüntü
     * @param mode Tespit modu ("E", "S", veya "M")
     * @return DetectionResponse veya null (hata durumunda)
     */
    suspend fun detectObjects(
        bitmap: Bitmap,
        mode: String = "E"
    ): Result<DetectionResponse> {
        return try {
            // Bitmap'i Base64'e çevir
            val base64Image = ImageUtils.bitmapToBase64(bitmap, quality = 85)
            
            // Request oluştur
            val request = DetectionRequest(
                image = base64Image,
                mode = mode
            )
            
            // API çağrısı
            val response = apiService.detectObjects(apiKey, request)
            
            // Response kontrolü
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                Result.failure(
                    Exception("API hatası: ${response.code()} - ${response.message()}")
                )
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

---

## Hata Yönetimi

### ApiErrorHandler.kt

```kotlin
package com.yourpackage.utils

import retrofit2.Response

object ApiErrorHandler {
    
    /**
     * API response'unu kontrol et ve hata mesajı döndür
     */
    fun <T> handleResponse(response: Response<T>): Result<T> {
        return when {
            response.isSuccessful && response.body() != null -> {
                Result.success(response.body()!!)
            }
            response.code() == 401 -> {
                Result.failure(Exception("Yetkilendirme hatası: Geçersiz API anahtarı"))
            }
            response.code() == 400 -> {
                Result.failure(Exception("Geçersiz istek: ${response.message()}"))
            }
            response.code() == 503 -> {
                Result.failure(Exception("Servis kullanılamıyor: Model yüklenemedi"))
            }
            response.code() == 500 -> {
                Result.failure(Exception("Sunucu hatası: ${response.message()}"))
            }
            else -> {
                Result.failure(Exception("Bilinmeyen hata: ${response.code()} - ${response.message()}"))
            }
        }
    }
}
```

### Güncellenmiş Repository (Hata Yönetimi ile)

```kotlin
package com.yourpackage.repository

import com.yourpackage.api.RetrofitClient
import com.yourpackage.model.DetectionRequest
import com.yourpackage.model.DetectionResponse
import com.yourpackage.utils.ApiErrorHandler
import com.yourpackage.utils.ImageUtils
import android.graphics.Bitmap

class VisionRepository {
    
    private val apiService = RetrofitClient.apiService
    private val apiKey = "your-secret-api-key-here"
    
    suspend fun detectObjects(
        bitmap: Bitmap,
        mode: String = "E"
    ): Result<DetectionResponse> {
        return try {
            // Bitmap'i Base64'e çevir
            val base64Image = ImageUtils.bitmapToBase64(bitmap, quality = 85)
            
            if (base64Image.isEmpty()) {
                return Result.failure(Exception("Görüntü dönüştürülemedi"))
            }
            
            // Request oluştur
            val request = DetectionRequest(
                image = base64Image,
                mode = mode
            )
            
            // API çağrısı
            val response = apiService.detectObjects(apiKey, request)
            
            // Hata yönetimi ile response kontrolü
            ApiErrorHandler.handleResponse(response)
            
        } catch (e: java.net.UnknownHostException) {
            Result.failure(Exception("İnternet bağlantısı yok"))
        } catch (e: java.net.SocketTimeoutException) {
            Result.failure(Exception("Bağlantı zaman aşımı"))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

---

## Tam Örnek Kod

### ViewModel Örneği

```kotlin
package com.yourpackage.ui

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yourpackage.repository.VisionRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class VisionViewModel : ViewModel() {
    
    private val repository = VisionRepository()
    
    private val _detectionState = MutableStateFlow<DetectionState>(DetectionState.Idle)
    val detectionState: StateFlow<DetectionState> = _detectionState
    
    fun detectObjects(bitmap: Bitmap, mode: String = "E") {
        viewModelScope.launch {
            _detectionState.value = DetectionState.Loading
            
            repository.detectObjects(bitmap, mode)
                .onSuccess { response ->
                    _detectionState.value = DetectionState.Success(response)
                }
                .onFailure { error ->
                    _detectionState.value = DetectionState.Error(error.message ?: "Bilinmeyen hata")
                }
        }
    }
}

sealed class DetectionState {
    object Idle : DetectionState()
    object Loading : DetectionState()
    data class Success(val response: DetectionResponse) : DetectionState()
    data class Error(val message: String) : DetectionState()
}
```

### Activity/Fragment Kullanımı

```kotlin
import android.graphics.Bitmap
import android.widget.Toast
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    private lateinit var viewModel: VisionViewModel
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        viewModel = ViewModelProvider(this)[VisionViewModel::class.java]
        
        // ViewModel state'i dinle
        lifecycleScope.launch {
            viewModel.detectionState.collect { state ->
                when (state) {
                    is DetectionState.Loading -> {
                        // Loading göster
                        showLoading(true)
                    }
                    is DetectionState.Success -> {
                        // Sonuçları göster
                        showLoading(false)
                        displayResults(state.response)
                    }
                    is DetectionState.Error -> {
                        // Hata mesajı göster
                        showLoading(false)
                        Toast.makeText(
                            this@MainActivity,
                            "Hata: ${state.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                    is DetectionState.Idle -> {
                        showLoading(false)
                    }
                }
            }
        }
        
        // Örnek: Kamera'dan görüntü al ve tespit et
        buttonDetect.setOnClickListener {
            val bitmap = getBitmapFromCamera() // Kendi implementasyonunuz
            viewModel.detectObjects(bitmap, mode = "E")  // Ev modu
        }
    }
    
    private fun displayResults(response: DetectionResponse) {
        // Tespit edilen nesneleri göster
        response.detections.forEach { detection ->
            println("${detection.label}: ${detection.score}")
            println("Konum: (${detection.boxPixels.x1}, ${detection.boxPixels.y1}) - " +
                    "(${detection.boxPixels.x2}, ${detection.boxPixels.y2})")
        }
        
        // Görüntü üzerine bounding box'ları çiz
        drawBoundingBoxes(response)
    }
    
    /**
     * API'den dönen box_pixels değerlerini kullanarak görüntü üzerine çiz
     */
    private fun drawBoundingBoxes(response: DetectionResponse) {
        // Örnek: ImageView üzerine çizim
        val imageView = findViewById<ImageView>(R.id.imageView)
        val bitmap = (imageView.drawable as BitmapDrawable).bitmap
        
        // Canvas oluştur
        val canvasBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(canvasBitmap)
        
        val paint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 32f
            typeface = Typeface.DEFAULT_BOLD
        }
        
        // Her tespit için box çiz
        response.detections.forEach { detection ->
            val box = detection.boxPixels
            
            // box_pixels değerleri doğrudan kullanılabilir (API'den pixel cinsinden geliyor)
            val rect = RectF(
                box.x1.toFloat(),
                box.y1.toFloat(),
                box.x2.toFloat(),
                box.y2.toFloat()
            )
            
            // Box çiz
            canvas.drawRect(rect, paint)
            
            // Label yaz
            val label = "${detection.label} (${String.format("%.2f", detection.score)})"
            canvas.drawText(label, box.x1.toFloat(), box.y1.toFloat() - 10, textPaint)
        }
        
        // Güncellenmiş bitmap'i göster
        imageView.setImageBitmap(canvasBitmap)
    }
    
    /**
     * Mod seçimi örneği
     */
    private fun selectMode(mode: String) {
        // Mode: "E" (Ev), "S" (Sokak), "M" (Market/Ofis)
        when (mode) {
            "E" -> {
                // Ev modu seçildi
                viewModel.detectObjects(getBitmapFromCamera(), mode = "E")
            }
            "S" -> {
                // Sokak modu seçildi
                viewModel.detectObjects(getBitmapFromCamera(), mode = "S")
            }
            "M" -> {
                // Market/Ofis modu seçildi
                viewModel.detectObjects(getBitmapFromCamera(), mode = "M")
            }
        }
    }
}
```

---

## Önemli Notlar

1. **API Key Güvenliği**: API anahtarınızı `strings.xml` veya `BuildConfig` içinde saklayın, kod içine yazmayın.

2. **Base64 NO_WRAP**: API'ye gönderirken mutlaka `Base64.NO_WRAP` kullanın.

3. **Görüntü Kalitesi**: Yüksek çözünürlüklü görüntüler için `quality` parametresini 70-85 arasında tutun.

4. **Network Thread**: Retrofit coroutines kullandığı için otomatik olarak background thread'de çalışır.

5. **Hata Yönetimi**: Tüm hata durumlarını (401, 400, 503, 500) kullanıcıya anlamlı mesajlarla gösterin.

6. **Timeout Ayarları**: Büyük görüntüler için timeout sürelerini artırın (30 saniye yeterli).

---

## Test Örneği

```kotlin
// Test için basit bir örnek
val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)
viewModel.detectObjects(bitmap, mode = "E")
```

---

## Sorun Giderme

### 401 Unauthorized
- API anahtarınızı kontrol edin
- Header'da `X-API-KEY` doğru gönderiliyor mu?

### 400 Bad Request
- Base64 string boş mu?
- Mode parametresi doğru mu? ("E", "S", veya "M")

### 503 Service Unavailable
- Model yüklenmemiş olabilir
- Sunucu kaynakları yetersiz olabilir

### Timeout
- Görüntü çok büyük olabilir
- İnternet bağlantısını kontrol edin

---

## İletişim

Sorularınız için: [API Dokümantasyonu](https://your-api-docs.com)

