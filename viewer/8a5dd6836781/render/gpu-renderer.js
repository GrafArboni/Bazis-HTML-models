// Уровень GI режима рендера (RF-039): WebGPU + SSGI на three.js 0.186.
//
// Файл не встраивается в страницу: выпуск кладёт папку gpu/ модуля в
// viewer/<версия>/render/ сайта (integration.site_files), страница подгружает её
// динамическим import() по кнопке «GI». Своя копия модели — только для этого холста:
// геометрия и текстуры берутся из сцены r160 (тот же буфер вершин, то же изображение),
// матрицы и видимость — каждый кадр, поэтому скрытие деталей, фасады, разнос сборщика и
// правки материалов видны сразу. Материалы — свойства исходных (их уже выставил
// базовый уровень по типу: roughness, metalness, envMapIntensity), цвет, текстура,
// прозрачность стекла.
//
// Основа — прототип Codex (outputs/webgpu-preview/gpu-renderer.js, 18–19.10.2026): тот же
// конвейер SSGI + TRAA и пересчёт камеры мм → м. Без редактора и каталога карт и без
// зеркал-отражателей (вторая очередь). Сторонний код — vendor/ (three.js, MIT, LICENSE).
import * as T from './vendor/three.webgpu.js';
import {
  pass,
  mrt,
  output,
  normalView,
  diffuseColor,
  velocity,
  vec4,
  mix,
  packNormalToRGB,
  unpackRGBToNormal,
  sample
} from './vendor/three.tsl.js';
import { ssgi } from './vendor/tsl/display/SSGINode.js';
import { traa } from './vendor/tsl/display/TRAANode.js';
import { RoomEnvironment } from './vendor/environments/RoomEnvironment.js';

const GROUPS = ['panels', 'profiles', 'imports', 'hardware'],
  MM = 0.001; // мм БАЗИСа → метры: освещение и SSGI считаются в метрах

export async function createGIRenderer(view) {
  const { scene: source, camera: sourceCamera, groups, canvas: host, isVisible, onError } = view;
  if (!navigator.gpu) throw new Error('в этом браузере нет WebGPU');
  const renderer = new T.WebGPURenderer({ antialias: false, preserveDrawingBuffer: true });
  try {
    await renderer.init();
  } catch (error) {
    renderer.dispose();
    throw new Error('WebGPU не запустился: ' + error.message);
  }
  if (!renderer.backend.isWebGPUBackend) {
    renderer.dispose();
    throw new Error('видеокарта не дала WebGPU');
  }
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFShadowMap; // PCFSoft в WebGPU-рендерере three нет
  const canvas = renderer.domElement;
  canvas.className = 'renderGpuCanvas';
  canvas.hidden = true;
  canvas.setAttribute('aria-hidden', 'true');
  host.after(canvas);
  renderer.onDeviceLost = () => onError?.(new Error('потеряна связь с видеокартой'));

  const scene = new T.Scene(),
    model = new T.Group(),
    geometries = new Map(),
    materials = new Map(),
    textures = new Map(),
    objects = new Map();
  model.scale.setScalar(MM);
  scene.add(model);

  function texture(src) {
    if (!src) return null;
    let result = textures.get(src);
    if (!result) {
      result = src.isDataTexture
        ? new T.DataTexture(src.image.data, src.image.width, src.image.height, src.format, src.type)
        : new T.Texture(src.image);
      for (const key of ['wrapS', 'wrapT', 'magFilter', 'minFilter', 'anisotropy', 'flipY', 'colorSpace', 'rotation'])
        result[key] = src[key];
      result.repeat.copy(src.repeat);
      result.offset.copy(src.offset);
      result.center.copy(src.center);
      result.needsUpdate = true;
      textures.set(src, result);
    }
    if (!src.isDataTexture && result.image !== src.image) {
      result.image = src.image;
      result.needsUpdate = true;
    }
    return result;
  }
  function material(src) {
    if (Array.isArray(src)) return src.map(material);
    let result = materials.get(src);
    const glass = !!src.userData?.glass;
    if (!result) {
      result = new T.MeshStandardMaterial({ side: src.side, transparent: glass, depthWrite: !glass });
      materials.set(src, result);
    }
    const map = texture(src.map);
    if (!!result.map !== !!map) result.needsUpdate = true;
    result.map = map;
    if (src.color) result.color.copy(src.color);
    result.roughness = src.roughness ?? 0.7;
    result.metalness = src.metalness ?? 0;
    result.envMapIntensity = src.envMapIntensity ?? 1;
    result.opacity = glass ? src.opacity : 1;
    return result;
  }
  function geometry(src) {
    let result = geometries.get(src);
    if (!result) {
      result = new T.BufferGeometry();
      for (const [name, attr] of Object.entries(src.attributes)) {
        if (attr.isInterleavedBufferAttribute) {
          const data = new attr.array.constructor(attr.count * attr.itemSize);
          for (let i = 0; i < attr.count; i++)
            for (let j = 0; j < attr.itemSize; j++)
              data[i * attr.itemSize + j] = attr.data.array[i * attr.data.stride + attr.offset + j];
          result.setAttribute(name, new T.BufferAttribute(data, attr.itemSize, attr.normalized));
        } else result.setAttribute(name, new T.BufferAttribute(attr.array, attr.itemSize, attr.normalized));
      }
      if (src.index) result.setIndex(new T.BufferAttribute(src.index.array, 1));
      for (const g of src.groups) result.addGroup(g.start, g.count, g.materialIndex);
      result.setDrawRange(src.drawRange.start, src.drawRange.count);
      if (!result.attributes.normal) result.computeVertexNormals();
      geometries.set(src, result);
    }
    return result;
  }
  function syncModel() {
    source.updateMatrixWorld();
    const present = new Set();
    for (const name of GROUPS)
      groups?.[name]?.traverse(src => {
        if (!src.isMesh || src.isBatchedMesh || src.userData?.decorative || !src.geometry?.attributes.position) return;
        present.add(src);
        let mesh = objects.get(src);
        if (!mesh) {
          mesh = new T.Mesh(geometry(src.geometry), material(src.material));
          mesh.matrixAutoUpdate = false;
          objects.set(src, mesh);
          model.add(mesh);
        }
        mesh.geometry = geometry(src.geometry);
        mesh.material = material(src.material);
        const glass = [].concat(src.material).every(m => m?.userData?.glass);
        mesh.castShadow = look.floor && !glass;
        mesh.receiveShadow = look.floor;
        mesh.matrix.copy(src.matrixWorld);
        mesh.matrixWorldNeedsUpdate = true;
        mesh.visible = isVisible(src);
      });
    for (const [src, mesh] of objects)
      if (!present.has(src)) {
        model.remove(mesh);
        objects.delete(src);
      }
    model.updateMatrixWorld(true);
  }

  // ---- свет, пол, фон ----
  let look = { exposure: 1, light: 2.4, floor: true, floorOpacity: 0.3, background: ['#f6f8f9', '#d3d9dd'], sun: [0.3, 0.75, 0.6], ...view.look };
  const sun = new T.DirectionalLight(0xfff4e6, look.light),
    // пол — настоящая поверхность цвета низа фона: от него свет отражается на детали (GI)
    floor = new T.Mesh(new T.PlaneGeometry(1, 1), new T.MeshStandardMaterial({ roughness: 0.95, metalness: 0, envMapIntensity: 0.3 }));
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0004;
  sun.shadow.radius = 3;
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(sun, sun.target, floor, new T.HemisphereLight(0xf0f8ff, 0x46505a, 0.3));
  const room = new RoomEnvironment(),
    pmrem = new T.PMREMGenerator(renderer),
    environment = pmrem.fromScene(room);
  scene.environment = environment.texture;
  room.dispose();
  pmrem.dispose();
  let backgroundTexture = null;
  function setBackground([top, bottom]) {
    backgroundTexture?.dispose();
    backgroundTexture = null;
    if (top === bottom) {
      scene.background = new T.Color(top);
      return;
    }
    const c = document.createElement('canvas');
    c.width = 2;
    c.height = 256;
    const g = c.getContext('2d'),
      fill = g.createLinearGradient(0, 0, 0, 256);
    fill.addColorStop(0, top);
    fill.addColorStop(1, bottom);
    g.fillStyle = fill;
    g.fillRect(0, 0, 2, 256);
    backgroundTexture = new T.CanvasTexture(c);
    backgroundTexture.colorSpace = T.SRGBColorSpace;
    scene.background = backgroundTexture;
  }
  let span = 1;
  function placeRig() {
    const box = new T.Box3();
    for (const mesh of objects.values()) if (mesh.visible) box.expandByObject(mesh);
    if (box.isEmpty()) return false;
    const center = box.getCenter(new T.Vector3()),
      radius = Math.max(box.getBoundingSphere(new T.Sphere()).radius, 0.05),
      cam = sun.shadow.camera;
    span = radius;
    sun.position.copy(center).addScaledVector(new T.Vector3(...look.sun).normalize(), radius * 3);
    sun.target.position.copy(center);
    Object.assign(cam, { left: -radius, right: radius, top: radius, bottom: -radius, near: radius * 0.5, far: radius * 6 });
    cam.updateProjectionMatrix();
    sun.shadow.normalBias = radius * 0.0015;
    floor.scale.set(radius * 12, radius * 12, 1);
    floor.position.set(center.x, box.min.y - 0.0005, center.z);
    return true;
  }
  function setLook(next) {
    look = { ...look, ...next };
    renderer.toneMappingExposure = look.exposure;
    sun.intensity = look.light * 1.4; // без света шаблона: солнце — главный источник
    sun.castShadow = look.floor;
    floor.visible = look.floor;
    floor.material.color.set(look.background[1]);
    setBackground(look.background);
    placeRig();
  }

  // ---- конвейер: проход сцены с MRT → SSGI → TRAA ----
  let camera, pipeline, gi, temporal, scenePass, ortho = null, width = 0, height = 0, disposed = false;
  function makePipeline(orthographic) {
    temporal?.dispose();
    gi?.dispose();
    scenePass?.dispose();
    pipeline?.dispose();
    ortho = orthographic;
    camera = orthographic ? new T.OrthographicCamera() : new T.PerspectiveCamera();
    scenePass = pass(scene, camera);
    scenePass.setMRT(mrt({ output, diffuseColor: vec4(diffuseColor.rgb, 0), normal: packNormalToRGB(normalView), velocity }));
    scenePass.getTexture('diffuseColor').type = T.UnsignedByteType;
    scenePass.getTexture('normal').type = T.UnsignedByteType;
    const color = scenePass.getTextureNode('output'),
      diffuse = scenePass.getTextureNode('diffuseColor'),
      depth = scenePass.getTextureNode('depth'),
      motion = scenePass.getTextureNode('velocity'),
      normals = sample(uv => unpackRGBToNormal(scenePass.getTextureNode('normal').sample(uv)));
    gi = ssgi(color, depth, normals, camera);
    gi.sliceCount.value = 2;
    gi.stepCount.value = 8;
    gi.radius.value = Math.min(span * 0.3, 1);
    gi.thickness.value = 0.018;
    gi.giIntensity.value = 3;
    gi.aoIntensity.value = 1.3;
    const lit = color.rgb.mul(gi.getAONode()).add(diffuse.rgb.mul(gi.getGINode().rgb));
    temporal = traa(vec4(mix(lit, color.rgb, diffuse.a), color.a), depth, motion, camera);
    pipeline = new T.RenderPipeline(renderer);
    pipeline.outputNode = temporal;
  }
  // Камера — копия камеры просмотрщика в метрах; параллельная проекция шаблона — та же
  // камера с ортографической матрицей (elements[15] === 1)
  function syncCamera() {
    const src = sourceCamera,
      orthographic = src.projectionMatrix.elements[15] === 1;
    if (ortho !== orthographic) makePipeline(orthographic);
    camera.position.copy(src.position).multiplyScalar(MM);
    camera.quaternion.copy(src.quaternion);
    camera.near = (orthographic ? -src.far : src.near) * MM;
    camera.far = src.far * MM;
    camera.projectionMatrix.copy(src.projectionMatrix);
    const e = camera.projectionMatrix.elements;
    if (orthographic) {
      camera.left = ((-1 - e[12]) / e[0]) * MM;
      camera.right = ((1 - e[12]) / e[0]) * MM;
      camera.bottom = ((-1 - e[13]) / e[5]) * MM;
      camera.top = ((1 - e[13]) / e[5]) * MM;
      e[0] /= MM;
      e[5] /= MM;
      e[10] /= MM;
    } else {
      e[14] *= MM;
      camera.aspect = src.aspect;
      camera.fov = src.fov;
    }
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.updateMatrixWorld(true);
  }
  function frame(scale = 1) {
    const w = Math.max(1, host.clientWidth),
      h = Math.max(1, host.clientHeight),
      ratio = Math.min(window.devicePixelRatio || 1, 1.5) * scale;
    if (w !== width || h !== height || renderer.getPixelRatio() !== ratio) {
      width = w;
      height = h;
      renderer.setPixelRatio(ratio);
      renderer.setSize(w, h, false);
    }
    syncModel();
    syncCamera();
    pipeline.render();
  }

  try {
    syncModel();
    if (!placeRig()) throw new Error('нет видимой геометрии для рендера');
    setLook(look);
  } catch (error) {
    renderer.dispose();
    canvas.remove();
    throw error;
  }
  let sweep = 0;
  return {
    canvas,
    render() {
      if (disposed) return;
      frame();
      if (++sweep % 30 === 0) placeRig(); // разнос сборщика, скрытие — пол и тень за ними
    },
    show(on) {
      canvas.hidden = !on;
    },
    setLook,
    // PNG текущего ракурса в scale раз больше окна
    async capture(scale = 2) {
      frame(scale);
      const url = canvas.toDataURL('image/png');
      width = 0;
      frame();
      return url;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      temporal?.dispose();
      gi?.dispose();
      scenePass?.dispose();
      pipeline?.dispose();
      for (const cache of [geometries, materials, textures]) for (const value of cache.values()) value.dispose();
      backgroundTexture?.dispose();
      environment.dispose();
      floor.geometry.dispose();
      floor.material.dispose();
      sun.shadow.dispose?.();
      renderer.dispose();
      canvas.remove();
    }
  };
}
