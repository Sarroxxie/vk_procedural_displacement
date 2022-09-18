// constants
//float triangleSize = 0.25;
//float pi = 3.1415926;
//mat2 latticeToWorld = mat2(triangleSize * cos(pi / 3), triangleSize, triangleSize * sin(pi / 3), 0);
//mat2 worldToLattice = inverse(latticeToWorld);

// Associates a random offset with vertex of triangle grid
// see "Procedural Textures by Tiling and Blending" Listing 1.3
vec2 hash(vec2 vertex) {
  return fract(sin((vertex) * mat2(127.1, 311.7, 269.5, 183.3)) * 43758.5453);
}

vec3 proceduralTilingAndBlending(vec2 uv, sampler2D inputTexture, float offset) {
  float w1, w2, w3;
  ivec2 vertex1 , vertex2 , vertex3;

  float triangleSize = 0.25;
  float pi = 3.1415926;
  mat2 latticeToWorld = mat2(triangleSize * cos(pi / 3), triangleSize, triangleSize * sin(pi / 3), 0);
  mat2 worldToLattice = inverse(latticeToWorld);

  vec2 pos = offset + (uv - 0.5); // could add a "*scale" here for scaling
  vec2 latticeCoord = worldToLattice * pos;
  ivec2 cell = ivec2(floor(latticeCoord));
  vec2 temp = fract(latticeCoord);

  // determining vertex coordinates in lattice
  vertex1 = cell;
  if (temp.x + temp.y >= 1.0) {
    vertex1 += 1;
    temp = 1.0 - temp.yx;
  }
  vertex2 = cell + ivec2(1,0);
  vertex3 = cell + ivec2(0,1);

  // setting weights
  w1 = 1.0 - temp.x - temp.y;
  w2 = temp.x;
  w3 = temp.y;

  // make sure to sample so that the hexagon fully lies within texture bounds
  vec2 uv1 = triangleSize + hash(vertex1) * (1 - 2 * triangleSize) + (pos - latticeToWorld * vertex1);
  vec2 uv2 = triangleSize + hash(vertex2) * (1 - 2 * triangleSize) + (pos - latticeToWorld * vertex2);
  vec2 uv3 = triangleSize + hash(vertex3) * (1 - 2 * triangleSize) + (pos - latticeToWorld * vertex3);

  // Sample Texture
  vec3 I1 = textureLod(inputTexture, uv1, 0).rgb;
  vec3 I2 = textureLod(inputTexture, uv2, 0).rgb;
  vec3 I3 = textureLod(inputTexture, uv3, 0).rgb;

  // Variance preserving blending
  vec3 G = w1 * I1 + w2 * I2 + w3 * I3;
  G = G - vec3(0.5);
  G = G * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
  G = G + vec3(0.5);

  //return vec3(hash(vertex1).x, hash(vertex2).x, hash(vertex3).x);

  //return vec3(floor(latticeCoord), 0);
  
  return G;
}