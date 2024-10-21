#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float u_min;
uniform float u_max;


uniform sampler2D heightMap;

void main()
{
	float height = texture(heightMap, aTexCoord).r / (u_max - u_min);
	vec3 displaced_position = vec3(aPos.x, height, aPos.z);
	gl_Position = projection * view * model * vec4(displaced_position, 1.0f);
}