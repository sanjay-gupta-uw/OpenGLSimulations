#include "ocean.h"
#include <stb_image.h>

using namespace glm;
using namespace std;

const int L = 1000; // Patch size

Ocean::Ocean(unsigned int SUBDIVISIONS)
{
   N = SUBDIVISIONS + 1;

   // initialize the model transformation matrices
   quad_model = glm::translate(quad_model, glm::vec3(0.0f, 2.0f, 0.0f));
   // quad_model = glm::rotate(quad_model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
   // quad_model = glm::scale(quad_model, glm::vec3(5.0f));

   ocean_model = glm::translate(ocean_model, glm::vec3(0.0f, -3.0f, 0.0f));

   genSurface();
   genWaveVectorField();
   initBuffers();
   cout << "Ocean initialized" << endl;
}

void Ocean::genSurface()
{
   // We will use a flat array to store the surface points
   surface = std::vector<float>(N * N * 5); // 5 floats per point (x, y, z, u, v)

   int SUBDIVISIONS = N - 1;
   float step_size = 2.0f / SUBDIVISIONS;

   // Initialize the grid points
   for (int i = 0; i < N; i++)
   {
      for (int j = 0; j < N; j++)
      {
         int index = (i * N + j) * 3;

         // Assign x, y positions, and z to 0 (initially flat)
         surface[index + 0] = -1.0f + i * step_size; // x
         surface[index + 1] = 0.0f;                  // y (height is initially 0)
         surface[index + 2] = -1.0f + j * step_size; // z

         // Assign texture coordinates
         surface[index + 3] = float(i) / SUBDIVISIONS;
         surface[index + 4] = float(j) / SUBDIVISIONS;
      }
   }
}

void Ocean::genWaveVectorField()
{
   K_ = new glm::vec2[N * N];

   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < N; ++j)
      {

         float n = int(j - (N / 2));
         float m = int(i - (N / 2));

         float Kx = 2.0f * glm::pi<float>() * n / L;
         float Ky = 2.0f * glm::pi<float>() * m / L;

         // cout << "Kx: " << Kx << " Ky: " << Ky << endl;

         K_[(i * N) + j] = glm::vec2(Kx, Ky);
      }
   }
}

void Ocean::initBuffers()
{
   ocean_shader = new Shader("ocean.vs", "ocean.fs");
   texture_shader = new Shader("texture.vs", "texture.fs");

   // Generate the VAO and VBO
   initQuadBuffers();
   initSurfaceBuffers();

   // Used for the height map
   glBindTexture(GL_TEXTURE_2D, heightTexture);
   // set the texture parameters
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

   // allocate memory for h0_k_
   h0_k_ = new std::complex<float>[N * N];
   h_kt_ = new std::complex<float>[N * N];

   generatePhillipsSpectrum();
}

void Ocean::initSurfaceBuffers()
{
   // Generate the VAO and VBO
   glGenVertexArrays(1, &OCEAN_VAO);
   glGenBuffers(1, &OCEAN_VBO);
   // glGenBuffers(1, &EBO);

   glBindVertexArray(OCEAN_VAO);

   glBindBuffer(GL_ARRAY_BUFFER, OCEAN_VBO);
   glBufferData(GL_ARRAY_BUFFER, surface.size() * sizeof(float), surface.data(), GL_STATIC_DRAW);

   // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
   // glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

   // Position attribute
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
   glEnableVertexAttribArray(0);
   // Texture attribtute
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
   glEnableVertexAttribArray(1);

   // Unbind the VAO
   glBindVertexArray(0);
}

void Ocean::initQuadBuffers()
{
   float squareVertices[] = {
       // Positions      // Texture Coords
       -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
       1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
       1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
       -1.0f, 1.0f, 0.0f, 0.0f, 1.0f};

   // Generate the VAO and VBO
   glGenVertexArrays(1, &QUAD_VAO);
   glGenBuffers(1, &QUAD_VBO);
   // glGenBuffers(1, &EBO);
   glGenTextures(1, &heightTexture);

   glBindVertexArray(QUAD_VAO);

   glBindBuffer(GL_ARRAY_BUFFER, QUAD_VBO);
   glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
   // Position attribute
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
   glEnableVertexAttribArray(0);
   // Texture attribtute
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
   glEnableVertexAttribArray(1);

   // Unbind the VAO
   glBindVertexArray(0);
}

void Ocean::setSharedUniforms(Shader *shader)
{
   shader->use();

   // Set transformation matrices
   shader->setMat4("view", camera->GetViewMatrix());
   shader->setMat4("projection", camera->GetProjectionMatrix());

   // Set texture sampler and min/max values for displacement
   shader->setInt("heightMap", 0); // Texture unit 0
   shader->setFloat("u_min", min_value);
   shader->setFloat("u_max", max_value);
}

void Ocean::update()
{
   generateH_KT_Spectrum(timeStep);
   generateHeightField();

   currentFrame = static_cast<float>(glfwGetTime());
   // refer to learnopengl.com for correct implementation of deltaTime to ensure consistent movement on machines with different refresh rates
   deltaTime = currentFrame - lastFrame;
   timeStep += deltaTime;
   // update Ocean
   // Take h0_k_ and generate time dependent component, h_kt
   // generateH_KT_Spectrum(timeStep);
   // generateHeightField();
   // cout << "ocean updated, preparing to render!" << endl;
   lastFrame = currentFrame;
}

void Ocean::renderQuad()
{
   setSharedUniforms(texture_shader);
   texture_shader->use(); // THIS SHOULD BE ENABLED FROM setSharedUniforms
   texture_shader->setMat4("model", quad_model);

   glBindVertexArray(QUAD_VAO);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, heightTexture);

   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDrawArrays(GL_TRIANGLE_FAN, 0, 4); // Draw the quad

   glBindVertexArray(0);
}

void Ocean::renderSurface()
{
   setSharedUniforms(ocean_shader);

   ocean_shader->use();
   ocean_shader->setMat4("model", ocean_model);
   glBindVertexArray(OCEAN_VAO);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, heightTexture);

   glPointSize(5.0f);
   glDrawArrays(GL_POINTS, 0, N * N); // Use the total number of points

   glBindVertexArray(0);
}

void Ocean::render(Camera &cam)
{
   camera = &cam;
   update();

   renderQuad();
   renderSurface();

   // render Quad

   // glActiveTexture(GL_TEXTURE0);
   // glBindTexture(GL_TEXTURE_2D, heightTexture);
   // shader->setInt("heightMap", 0);
   // // Set point size
   // glPointSize(5.0f); // Makes the points larger so you can see them
   // // Render the ocean
   // glBindVertexArray(VAO);
   // glDrawArrays(GL_POINTS, 0, N * N); // Use the total number of points
   // glBindVertexArray(0);
   // cout << "ocean rendered!" << endl;
   // }
}

Ocean::~Ocean()
{
   // Clean up
   glDeleteVertexArrays(1, &OCEAN_VAO);
   glDeleteBuffers(1, &OCEAN_VBO);
   glDeleteVertexArrays(1, &QUAD_VAO);
   glDeleteBuffers(1, &QUAD_VBO);
   glDeleteTextures(1, &heightTexture);
}

float Ocean::Phillips(glm::vec2 K)
{
   float K_mag = length(K); // since K is our Ocean vector

   if (K_mag < 0.00001f)
      return 0.0f;

   float L = v * v / G;
   float retVal = float(
       A * float(std::exp(-1.0f / std::pow(K_mag * L, 2))) *
       std::pow(glm::dot(glm::normalize(windDir), glm::normalize(K)), 2) /
       std::pow(K_mag, 4));
   if (retVal != retVal)
      cout << "NaN detected" << endl;
   return retVal;
}

void Ocean::generatePhillipsSpectrum()
{
   cout << "Generating Phillips Spectrum" << endl;

   std::random_device rd;
   std::mt19937 gen(rd());
   std::normal_distribution<float> distribution(0.0f, 1.0f);

   // Generate the Phillips spectrum
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < N; ++j)
      {
         float gauss_real = distribution(gen);
         float gauss_img = distribution(gen);
         int index = i * N + j;
         if (length(K_[index]) < 0.00001f)
         {
            // cout << "Mag is 0 at index i: " << i << " j: " << j << endl;
         }

         float P = std::sqrt(Phillips(K_[index]) * 0.5f);

         std::complex<float> h0 = std::complex<float>(gauss_real * P, gauss_img * P);

         // Assign the generated value
         h0_k_[index] = h0;
      }
   }

   cout << "Generated Phillips Spectrum with Hermitian symmetry" << endl;
}

// Should be computed on the GPU
void Ocean::generateH_KT_Spectrum(float t)
{
   // Generate h_kt from h0_k_
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < N; ++j)
      {
         // -K: N - i - 1, N - j - 1 (DESMOS: f\left(x\right)=\left(-x+N\right)-1)
         int K_index = i * N + j;
         int minus_K_index = (N - 1 - i) * N + (N - 1 - j);

         // Calculate the angular frequency w(k) (dispersion relation)
         float w_k = std::sqrt(G * length(K_[K_index]));

         // Calculate the time-dependent Fourier amplitudes
         std::complex<float> exp_iwt =
             std::exp(std::complex<float>(0, w_k * t)); // e^(i * w(k) * t)
         std::complex<float> exp_neg_iwt =
             std::exp(std::complex<float>(0, -w_k * t)); // e^(-i * w(k) * t)

         // Compute h_kt_ at this K
         h_kt_[K_index] = h0_k_[K_index] * exp_iwt + std::conj(h0_k_[minus_K_index]) * exp_neg_iwt;
      }
   }
}

/*
void Ocean::generateHeightField()
{
   // Create arrays to store the spatial domain height field
   fftwf_complex *h_kt_complex =
       fftwf_alloc_complex(N * N); // FFTW input (frequency domain)
   fftwf_complex *height_field =
       fftwf_alloc_complex(N * N); // FFTW output (spatial domain)

   // Copy h_kt_ (frequency domain) data to h_kt_complex
   for (int i = 0; i < N; ++i)
   {
      for (int j = 0; j < N; ++j)
      {
         int index = i * N + j;
         h_kt_complex[index][0] = h_kt_[index].real(); // Real part
         h_kt_complex[index][1] = h_kt_[index].imag(); // Imaginary part
      }
   }

   // Create an FFTW plan to compute the 2D IFFT
   fftwf_plan ifft_plan = fftwf_plan_dft_2d(N, N, h_kt_complex, height_field,
                                            FFTW_BACKWARD, FFTW_ESTIMATE);

   // Execute the IFFT
   fftwf_execute(ifft_plan);

   // Normalize the result (scaling after FFTW IFFT)
   float N2 = N * N;
   std::vector<float> real_part(N2); // Real part of the height field
   for (int i = 0; i < N2; ++i)
   {
      // height_field[i][0] /= (N2);  // Normalize real part
      // height_field[i][1] /= (N2);  // Normalize imaginary part (though it
      // should be very small)
      real_part[i] = height_field[i][0] / (N2); // Store the real part
   }

   // Free the FFTW plan and input/output arrays
   fftwf_destroy_plan(ifft_plan);
   fftwf_free(h_kt_complex);

   // glBindTexture(GL_TEXTURE_2D, heightMapTexture);

   // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RED, GL_FLOAT, real_part.data());
   // // get max and min from real_part
   // float max = *max_element(real_part.begin(), real_part.end());
   // float min = *min_element(real_part.begin(), real_part.end());
   // cout << "Max: " << max << " Min: " << min << endl;

   // Save the image
   cv::Mat heightFieldImage(N, N, CV_32FC1, real_part.data());
   cv::normalize(heightFieldImage, heightFieldImage, 0, 255, cv::NORM_MINMAX);
   cv::imwrite("heightField.png", heightFieldImage);

   glBindTexture(GL_TEXTURE_2D, heightTexture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, N, N, 0, GL_RED, GL_FLOAT, real_part.data());
   glGenerateMipmap(GL_TEXTURE_2D);

   // Free the height field
   fftwf_free(height_field);
   return;

   // Read image into texture
   // load and create a texture
   // --------------------------------
   glBindTexture(GL_TEXTURE_2D, heightTexture);
   // set the texture parameters
   // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_NEAREST);
   // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_NEAREST);
   // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   // load image, create texture and generate mipmaps
   int width, height, nrChannels;
   stbi_set_flip_vertically_on_load(true); // tell stb_image.h to flip loaded texture's on the y-axis.
   // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
   unsigned char *data = stbi_load("heightField.png", &width, &height, &nrChannels, 1);
   // unsigned char *data = stbi_load("../../resources/textures/awesomeface.png", &width, &height, &nrChannels, 1);
   if (data)
   {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data);
      glGenerateMipmap(GL_TEXTURE_2D);
   }
   else
   {
      std::cout << "Failed to load texture" << std::endl;
      exit(1);
   }
   stbi_image_free(data);
}
*/

void Ocean::generateHeightField()
{
   auto h_kt_complex = std::unique_ptr<fftwf_complex[], decltype(&fftwf_free)>(
       fftwf_alloc_complex(N * N), fftwf_free);
   auto height_field = std::unique_ptr<fftwf_complex[], decltype(&fftwf_free)>(
       fftwf_alloc_complex(N * N), fftwf_free);

   // Copy frequency domain data
   for (int index = 0; index < N * N; ++index)
   {
      h_kt_complex[index][0] = h_kt_[index].real();
      h_kt_complex[index][1] = h_kt_[index].imag();
   }

   // Create and execute the FFTW plan
   fftwf_plan ifft_plan = fftwf_plan_dft_2d(N, N, h_kt_complex.get(), height_field.get(),
                                            FFTW_BACKWARD, FFTW_ESTIMATE);
   if (!ifft_plan)
   {
      std::cerr << "Failed to create FFTW plan!" << std::endl;
      return;
   }

   fftwf_execute(ifft_plan);

   // Extract real part into a vector
   std::vector<float> real_part(N * N);
   for (int i = 0; i < N * N; ++i)
   {
      real_part[i] = height_field[i][0];
   }

   // Find min and max values
   auto [min_it, max_it] = std::minmax_element(real_part.begin(), real_part.end());
   min_value = *min_it;
   max_value = *max_it;

   // Upload the real part as a texture (without normalization)
   glBindTexture(GL_TEXTURE_2D, heightTexture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, N, N, 0, GL_RED, GL_FLOAT, real_part.data());
   glGenerateMipmap(GL_TEXTURE_2D);

   fftwf_destroy_plan(ifft_plan);
}
