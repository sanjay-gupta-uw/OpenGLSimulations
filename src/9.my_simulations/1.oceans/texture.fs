#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform float u_min;
uniform float u_max;

uniform sampler2D heightMap;

void main()
{
    float height = texture(heightMap, TexCoord).r;
    // Visualize the height value as grayscale color
    float normalizedHeight = (height - u_min) / (u_max - u_min);

    FragColor = vec4(vec3(normalizedHeight), 1.0);
}
