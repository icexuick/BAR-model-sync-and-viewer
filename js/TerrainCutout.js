import {
  BoxGeometry,
  Float32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  DoubleSide
} from "three";

class TerrainCutout extends Mesh {
  constructor(width, height, depth, segW, segD, heightMap) {
    super();

    this.geometry = new BoxGeometry(width, height, depth, segW, 1, segD);
    let pos = this.geometry.attributes.position;
    let nor = this.geometry.attributes.normal;
    let enableDisplacement = [];
    for (let i = 0; i < pos.count; i++) {
      enableDisplacement.push(
        Math.sign(pos.getY(i)), // point can be displaced
        Math.sign(nor.getY(i)) // normal needs to be re-computed
      );
      //re-compute UV (for displacement)
      let u = (pos.getX(i) + width * 0.5) / width;
      let v = 1 - (pos.getZ(i) + depth * 0.5) / depth;
      this.geometry.attributes.uv.setXY(i, u, v);
    }
    this.geometry.setAttribute(
      "enableDisp",
      new Float32BufferAttribute(enableDisplacement, 2)
    );
    this.material = new MeshStandardMaterial({
      //wireframe: true,
      //side: DoubleSide,
      color: "brown",
      displacementMap: heightMap,
      onBeforeCompile: (shader) => {
        shader.vertexShader = `
          attribute vec2 enableDisp;
          
          ${shader.vertexShader}
        `.replace(
          `#include <displacementmap_vertex>`,
          `
          #ifdef USE_DISPLACEMENTMAP
            if (enableDisp.x > 0.) {
              
              vec3 vUp = vec3(0, 1, 0);

              vec3 v0 = normalize( vUp ) * ( texture2D( displacementMap, vUv ).x * displacementScale + displacementBias );
              transformed += v0;
              
              if(enableDisp.y > 0.) {
                float txl = 1. / 256.;

                vec3 v1 = normalize( vUp ) * ( texture2D( displacementMap, vUv + vec2(txl, 0.) ).x * displacementScale + displacementBias );
                v1.xz = vec2(txl, 0.) * 20.;
                vec3 v2 = normalize( vUp ) * ( texture2D( displacementMap, vUv + vec2(0., txl) ).x * displacementScale + displacementBias );
                v2.xz = -vec2(0., txl) * 20.;

                vec3 n = normalize(cross(v1 - v0, v2 - v0));
                vNormal = normalMatrix * n;
              }              
            }
          #endif
          `
        );
        //console.log(shader.vertexShader);
      }
    });
  }
}

export { TerrainCutout };
