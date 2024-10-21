#ifndef OCEAN_H
#define OCEAN_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <opencv2/opencv.hpp>

#include <fftw3.h>
// #include "stb\stb_image_write.h"

#include <learnopengl/shader.h>
#include <learnopengl/camera.h>

class Ocean
{
   bool render_texture = true;
   float min_value;
   float max_value;

   // Rendering parameters
   unsigned int OCEAN_VBO, OCEAN_VAO, OCEAN_EBO;
   unsigned int QUAD_VBO, QUAD_VAO;
   std::vector<float> surface; // Vertices of the ocean surface
   std::vector<unsigned int> indices;
   unsigned int heightTexture;

   // model matrix
   glm::mat4 quad_model = glm::mat4(1.0f);
   glm::mat4 ocean_model = glm::mat4(1.0f);

   void genSurface();
   void genWaveVectorField();
   void initQuadBuffers();
   void initSurfaceBuffers();
   void initBuffers();
   void setSharedUniforms(Shader *shader);
   float Phillips(glm::vec2 K);
   void generatePhillipsSpectrum();
   void generateH_KT_Spectrum(float t);
   void generateHeightField();
   void update();

   void renderQuad();
   void renderSurface();

public:
   // Ocean parameters
   std::complex<float> *h0_k_;
   std::complex<float> *h_kt_;
   glm::vec2 *K_;

   unsigned int N;
   float A = 4.0f;
   float G = 9.81f;
   float v = 40.0f;
   glm::vec2 windDir = glm::vec2(1.0f, 1.0f);

   // Frame timing
   float currentFrame = 0.0f;
   float lastFrame = 0.0f;
   float deltaTime = 0.0f;
   float timeStep = 0.0f; // 't' value used for ocean simulation

   Camera *camera;

   Shader *ocean_shader;
   Shader *texture_shader;

   // Ocean functions
   Ocean(unsigned int SUBDIVISIONS);
   ~Ocean();
   void render(Camera &camera);
};

#endif // OCEAN_H