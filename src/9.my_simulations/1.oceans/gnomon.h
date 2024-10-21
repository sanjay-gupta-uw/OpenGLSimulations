#ifndef GNOMON_H
#define GNOMON_H

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <iostream>

class Gnomon
{
   unsigned int VBO, VAO, EBO;
   Shader *shader;

public:
   Gnomon()
   {
      shader = new Shader("gnomon.vs", "gnomon.fs");

      float gnomonVertices[] = {
          // Positions       // Colors
          0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // Origin (shared for all axes)
          1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X-axis endpoint
          0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y-axis endpoint
          0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f  // Z-axis endpoint
      };

      // Indices defining the lines (pairs of vertices)
      unsigned int gnomonIndices[] = {
          0, 1, // Line from origin to X-axis endpoint
          0, 2, // Line from origin to Y-axis endpoint
          0, 3  // Line from origin to Z-axis endpoint
      };

      glGenVertexArrays(1, &VAO);
      glGenBuffers(1, &VBO);
      glGenBuffers(1, &EBO);

      glBindVertexArray(VAO);

      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferData(GL_ARRAY_BUFFER, sizeof(gnomonVertices), gnomonVertices, GL_STATIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(gnomonIndices), gnomonIndices, GL_STATIC_DRAW);

      // Position attribute
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
      glEnableVertexAttribArray(0);

      // Color attribute
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
      glEnableVertexAttribArray(1);

      glBindVertexArray(0);
   }
   void render(Camera &camera)
   {
      shader->use();
      shader->setMat4("view", camera.GetViewMatrix());
      shader->setMat4("projection", camera.GetProjectionMatrix());
      glm::mat4 model = glm::mat4(1.0f);
      shader->setMat4("model", model);

      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      glBindVertexArray(VAO);
      glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, 0);
      // std::cout << "gnomon rendered" << std::endl;
   }
};

#endif // GNOMON_H