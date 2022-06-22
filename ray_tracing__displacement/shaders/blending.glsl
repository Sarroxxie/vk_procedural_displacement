// Associates a random offset with vertex of triangle grid
// see "Procedural Textures by Tiling and Blending" Listing 1.3
vec2 hash(vec2 pos) {
  return fract(sin((pos) * mat2(127.1, 311.7, 269.5, 183.3)) * 43758.5453);
}

// Compute local triangle barycentric coordinates and vertex IDs
void triangleGrid(vec2 uv, out float w1, out float w2, out float w3, out ivec2 vertex1, out ivec2 vertex2, out ivec2 vertex3)
{
  // Scaling of the input
  uv *= 3.464; // 2 * sqrt (3)

  // Skew input space into simplex triangle grid
  const mat2 gridToSkewedGrid = mat2 (1.0 , 0.0 , -0.57735027 , 1.15470054) ;
  vec2 skewedCoord = gridToSkewedGrid * uv;

  // Compute local triangle vertex IDs and local barycentric coordinates
  ivec2 baseId = ivec2(floor(skewedCoord));
  vec3 temp = vec3(fract(skewedCoord), 0);
  temp .z = 1.0 - temp.x - temp.y;
  if (temp.z > 0.0)
  {
    w1 = temp.z;
    w2 = temp.y;
    w3 = temp.x;
    vertex1 = baseId ;
    vertex2 = baseId + ivec2(0, 1);
    vertex3 = baseId + ivec2(1, 0);
  }
  else
  {
    w1 = -temp.z;
    w2 = 1.0 - temp.y;
    w3 = 1.0 - temp.x;
    vertex1 = baseId + ivec2(1, 1);
    vertex2 = baseId + ivec2(1, 0);
    vertex3 = baseId + ivec2(0, 1);
  }
}

vec3 proceduralTilingAndBlending(vec2 uv, sampler2D inputTexture) {
  // Get triangle info
  float w1 , w2 , w3;
  ivec2 vertex1 , vertex2 , vertex3;
  triangleGrid(uv, w1 , w2, w3, vertex1, vertex2, vertex3);

  // Assign random offset to each triangle vertex
  vec2 uv1 = uv + hash(vertex1);
  vec2 uv2 = uv + hash(vertex2);
  vec2 uv3 = uv + hash(vertex3);

  // Fetch input
  vec3 I1 = texture(inputTexture, uv1).rgb;
  vec3 I2 = texture(inputTexture, uv2).rgb;
  vec3 I3 = texture(inputTexture, uv3).rgb;

  // Linear blending
  vec3 G = w1 * I1 + w2 * I2 + w3 * I3;
  G = G - vec3(0.5);
  G = G * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
  G = G + vec3(0.5);
  return G;
}