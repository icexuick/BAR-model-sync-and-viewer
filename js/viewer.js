    import * as THREE from 'three';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
    import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
    import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
    import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';
    import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
    import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';
    import GUI from 'three/addons/libs/lil-gui.module.min.js';

    // --- CONFIGURATIE ---
    const VIEWER_CONFIG = {
        autoRotationSpeed: 0.22,
        mouseRotateSpeed: 0.5,
        envMapIntensity: 1.05,
        hdrUrl: '',
        toneMappingExposure: 1.5,
        roughnessOffset: 0.0,
        teamColor: '#0043EE',

        // Lichten
        keyLightIntensity: 0.7,
        keyLightColor: '#ffffff',
        keyLightX: 3, keyLightY: 6, keyLightZ: 2,
        shadowRadius: 8,
        shadowBlurSamples: 8,

        fillLightIntensity: 0.1,
        fillLightColor: '#ffffff',
        fillLightX: -3, fillLightY: 2, fillLightZ: -4,

        ambientLightIntensity: 0.02,
        ambientLightColor: '#ffffff',

        // PostFX
        ssaoEnabled: true,
        ssaoRadius: 16,
        ssaoMinDist: 0.001,
        ssaoMaxDist: 5.5,

        exportWidth: 4096,
        initialZoomMultiplier: 0.70,
        cameraHeightFactor: 0.8
    };

    function isMobileDevice() {
        const ua = navigator.userAgent.toLowerCase();
        return /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/.test(ua) ||
               ('maxTouchPoints' in navigator && navigator.maxTouchPoints > 0 && window.innerWidth < 768);
    }
    const isMobile = isMobileDevice();

    const performanceSettings = {
        pixelRatioCap: isMobile ? Math.min(window.devicePixelRatio, 2.0) : window.devicePixelRatio,
        shadowMapSize: isMobile ? 512 : 1024,
        shadowMapType: THREE.VSMShadowMap,
        anisotropy: isMobile ? 2 : 4,
        ssaoKernelSize: isMobile ? 6 : 8,
        ssaoKernelRadius: isMobile ? 8 : 16,
        ssaoEnabled: true,
        powerPreference: isMobile ? 'low-power' : 'high-performance',
    };

    // FIX: word-boundary matching zodat 'glow' niet matcht in 'legLowerL1' (le-glow-er).
    // Splitst camelCase namen in losse woorden en checkt elk keyword als heel woord.
    const HIDE_KEYWORDS = ['flare','fire','emit','jam','wake','bow','nano','blink','glow','light'];
    function shouldHideMesh(name) {
        const words = name.replace(/([A-Z])/g, ' $1').toLowerCase().split(/[\s_\-\.0-9]+/).filter(Boolean);
        return HIDE_KEYWORDS.some(k => words.includes(k));
    }

    window.addEventListener('load', function() {
    setTimeout(function() {
        const urlParams = new URLSearchParams(window.location.search);
        const isEditorMode = urlParams.has('editor');

        const container = document.getElementById('model-container');
        const animToggleButton = document.getElementById('anim-toggle-button');
        const rotateToggleButton = document.getElementById('rotate-toggle-button');
        const shadowToggleButton = document.getElementById('shadow-toggle-button');
        const exportButton = document.getElementById('export-button');
        const noticeElement = document.getElementById('model-viewer-notice');

        if (!container) return;
        container.style.display = 'block';

        // --- FADE-IN SETUP: Start onzichtbaar ---
        container.style.opacity = '0';
        container.style.transform = 'scale(0.5)';

        // --- UNIT NAME LOGIC ---
        let unitName = 'armflea';
        const nameField = document.getElementById('unit-name-text');
        if (nameField && nameField.textContent.trim() !== '') {
            unitName = nameField.textContent.trim();
        } else {
            const pathSegments = window.location.pathname.split('/');
            unitName = pathSegments.pop() || pathSegments.pop() || 'armflea';
            if (unitName.toLowerCase().endsWith('.html')) unitName = unitName.substring(0, unitName.length - 5);
        }
        unitName = unitName.replace('.glb', '').replace(/[^a-zA-Z0-9_-]/g, '');

        // --- FACTION SETTINGS ---
        let teamColorValue = 0x0043EE, texturePrefix = 'arm';
        let normalMapScaleY = 2.2, roughnessValue = 0.45;
        let pbrEmissiveIntensityValue = 14.0, pulseMaxValue = 1.4;

        const hdrBasePath = 'https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/hdr/';
        let armHdrUrl = hdrBasePath + 'clarens_midday_2k5.hdr';
        let corHdrUrl = hdrBasePath + 'clarens_midday_2k5.hdr';
        let legHdrUrl = hdrBasePath + 'clarens_midday_2k5.hdr';
        let selectedHdrUrl = armHdrUrl;

        if (unitName.startsWith('cor')) {
            teamColorValue = 0xFF0000; texturePrefix = 'cor';
            roughnessValue = 0.50; pbrEmissiveIntensityValue = 15.0;
            selectedHdrUrl = corHdrUrl; VIEWER_CONFIG.envMapIntensity = 0.95;
        } else if (unitName.startsWith('leg')) {
            teamColorValue = 0x00FF00; texturePrefix = 'leg';
            roughnessValue = 0.8; pbrEmissiveIntensityValue = 12.0;
            selectedHdrUrl = legHdrUrl; VIEWER_CONFIG.envMapIntensity = 0.89;
        }

        const teamColor = new THREE.Color(teamColorValue);
        VIEWER_CONFIG.teamColor = '#' + teamColor.getHexString();
        VIEWER_CONFIG.hdrUrl = selectedHdrUrl;

        const modelURL = `https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/${unitName}.glb?v=${Date.now()}`;
        const baseURL = 'https://raw.githubusercontent.com/icexuick/BAR-modelviewer/main/tex/';
        const diffuseURL = `${baseURL}${texturePrefix}_color.png`;
        const pbrURL = `${baseURL}${texturePrefix}_other.png`;
        const normalURL = `${baseURL}${texturePrefix}_normal.png`;
        const teamURL = `${baseURL}${texturePrefix}_team.png`;

        // --- Scene Setup ---
        const scene = new THREE.Scene();
        scene.environmentRotation = new THREE.Euler(0, 0, 0);
        scene.background = null;

        const camera = new THREE.PerspectiveCamera(35, container.offsetWidth / container.offsetHeight, 0.1, 10000);

        const renderer = new THREE.WebGLRenderer({
            antialias: false, alpha: true, preserveDrawingBuffer: true,
            powerPreference: performanceSettings.powerPreference
        });
        window.renderer = renderer;
        renderer.setPixelRatio(performanceSettings.pixelRatioCap);
        renderer.setSize(container.offsetWidth, container.offsetHeight);
        renderer.setClearColor(0x000000, 0);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = VIEWER_CONFIG.toneMappingExposure;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = performanceSettings.shadowMapType;
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true; controls.dampingFactor = 0.05;
        controls.enableZoom = true; controls.rotateSpeed = VIEWER_CONFIG.mouseRotateSpeed;
        controls.zoomSpeed = 1.2; controls.minDistance = 1; controls.maxDistance = 5000;

        const downloadButton = document.getElementById('download-glb-button');
        if (downloadButton) { downloadButton.href = modelURL; downloadButton.download = `${unitName}.glb`; }

        // --- Lights ---
        const lightGroup = new THREE.Group(); scene.add(lightGroup);

        const directionalLight = new THREE.DirectionalLight(VIEWER_CONFIG.keyLightColor, VIEWER_CONFIG.keyLightIntensity);
        directionalLight.position.set(VIEWER_CONFIG.keyLightX, VIEWER_CONFIG.keyLightY, VIEWER_CONFIG.keyLightZ);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.set(performanceSettings.shadowMapSize, performanceSettings.shadowMapSize);

        directionalLight.shadow.bias = -0.0005;

        directionalLight.shadow.radius = VIEWER_CONFIG.shadowRadius;
        directionalLight.shadow.blurSamples = VIEWER_CONFIG.shadowBlurSamples;
        lightGroup.add(directionalLight);

        const directionalLight2 = new THREE.DirectionalLight(VIEWER_CONFIG.fillLightColor, VIEWER_CONFIG.fillLightIntensity);
        directionalLight2.position.set(VIEWER_CONFIG.fillLightX, VIEWER_CONFIG.fillLightY, VIEWER_CONFIG.fillLightZ);
        lightGroup.add(directionalLight2);

        const ambientLight = new THREE.AmbientLight(VIEWER_CONFIG.ambientLightColor, VIEWER_CONFIG.ambientLightIntensity);
        scene.add(ambientLight);

        let material, modelScene, groundMesh, mixer = null, currentAction = null;
        const timeUniform = { value: 0.0 };
        const clock = new THREE.Clock();
        let isAltDown = false, isTurntableEnabledByUser = true, turntableActive = false, isInteracting = false, idleTimer = null;
        let areShadowsVisible = true;
        let composer, renderPass, ssaoPass, smaaPass, outputPass;

        const textureLoader = new THREE.TextureLoader();
        const gltfLoader = new GLTFLoader();
        const rgbeLoader = new RGBELoader();

        const loadHDR = (url) => {
            rgbeLoader.load(url, (texture) => {
                texture.mapping = THREE.EquirectangularReflectionMapping;
                scene.environment = texture; scene.background = null;
            }, undefined, (err) => { console.error("HDR Error:", err); alert("Kon HDR niet laden."); });
        };

        Promise.all([
            rgbeLoader.loadAsync(selectedHdrUrl).then(t => { t.mapping = THREE.EquirectangularReflectionMapping; scene.environment = t; scene.background = null; return t; }),
            textureLoader.loadAsync(diffuseURL), textureLoader.loadAsync(pbrURL),
            textureLoader.loadAsync(normalURL), textureLoader.loadAsync(teamURL),
            gltfLoader.loadAsync(modelURL)
        ]).then(([env, diffuseMap, pbrMap, normalMap, teamMap, gltf]) => {

            const fallbackImageBlock = document.querySelector('.flex-unit-detail-img');
            if (fallbackImageBlock) fallbackImageBlock.style.display = 'none';

            // --- FADE-IN TRIGGER: Model is geladen, fade in na delay ---
            setTimeout(() => {
                container.style.transition = 'opacity 0.45s ease-out, transform 0.7s ease-out';
                container.style.opacity = '1';
                container.style.transform = 'scale(1)';
            }, 900);

            const setupTex = (t, s) => { if(t) { t.colorSpace = s; t.flipY = false; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = performanceSettings.anisotropy; }};
            setupTex(diffuseMap, THREE.SRGBColorSpace); setupTex(pbrMap, THREE.LinearSRGBColorSpace);
            setupTex(normalMap, THREE.LinearSRGBColorSpace); setupTex(teamMap, THREE.LinearSRGBColorSpace);

            material = new THREE.MeshStandardMaterial({
                map: diffuseMap, normalMap: normalMap, normalScale: new THREE.Vector2(1, normalMapScaleY),
                metalness: 1.0, roughness: roughnessValue, envMapIntensity: VIEWER_CONFIG.envMapIntensity,
                side: THREE.DoubleSide, transparent: true
            });

            material.onBeforeCompile = (shader) => {
                shader.defines.USE_ENVMAP_ROTATION = '1';
                shader.uniforms.pbrMap = { value: pbrMap };
                shader.uniforms.teamColor = { value: teamColor };
                shader.uniforms.teamMap = { value: teamMap };
                shader.uniforms.pbrEmissiveIntensity = { value: pbrEmissiveIntensityValue };
                shader.uniforms.time = timeUniform;
                shader.uniforms.roughnessOffset = { value: VIEWER_CONFIG.roughnessOffset };

                if (!shader.uniforms.envMapRotation) shader.uniforms.envMapRotation = { value: new THREE.Matrix3() };

                shader.vertexShader = 'varying vec2 vUv;\n' + shader.vertexShader;
                shader.vertexShader = shader.vertexShader.replace('#include <uv_vertex>', '#include <uv_vertex>\n vUv = uv;');

                const header = [
                    'precision highp float;',
                    'varying vec2 vUv;',
                    'uniform sampler2D pbrMap;',
                    'uniform vec3 teamColor;',
                    'uniform sampler2D teamMap;',
                    'uniform float pbrEmissiveIntensity;',
                    'uniform float time;',
                    'uniform float roughnessOffset;',
                    '#define USE_TEAMMAP'
                ].join('\n');

                shader.fragmentShader = header + '\n' + shader.fragmentShader;

                shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>', `float roughnessFactor = roughness + roughnessOffset; vec4 texelRoughness = texture2D( pbrMap, vUv ); roughnessFactor *= texelRoughness.b; roughnessFactor = clamp(roughnessFactor, 0.05, 1.0);`);
                shader.fragmentShader = shader.fragmentShader.replace('#include <metalnessmap_fragment>', `float metalnessFactor = metalness; vec4 texelMetalness = texture2D( pbrMap, vUv ); metalnessFactor *= texelMetalness.g;`);
                shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>\n #ifdef USE_TEAMMAP\n float S_teamMask = texture2D( teamMap, vUv ).r; float S_blendFactor = step(0.35, S_teamMask); vec3 targetColor = mix(diffuseColor.rgb, teamColor, 0.75); diffuseColor.rgb = mix(diffuseColor.rgb, targetColor, S_blendFactor); \n #endif`);

                shader.fragmentShader = shader.fragmentShader.replace('#include <tonemapping_fragment>', `
                    vec3 S_pbrBoost = vec3(0.0);
                    #ifdef USE_MAP
                        float S_emissionAmount = texture2D( pbrMap, vUv ).r;
                        if (S_emissionAmount > 0.01) {
                            float S_pulse = (sin(time * 3.14159) * 0.5 + 0.5) * (${pulseMaxValue.toFixed(2)} - 0.01) + 0.01;
                            vec4 S_diff = texture2D( map, vUv );
                            S_pbrBoost = pow(S_diff.rgb, vec3(2.2)) * S_emissionAmount * pbrEmissiveIntensity * S_pulse;
                        }
                    #endif
                    vec3 S_combinedLight = outgoingLight + totalEmissiveRadiance + S_pbrBoost;
                    #ifdef USE_TONEMAPPING
                        gl_FragColor.rgb = toneMapping( S_combinedLight );
                    #else
                        gl_FragColor.rgb = S_combinedLight;
                    #endif
                    gl_FragColor.a = diffuseColor.a;
                `);
                shader.fragmentShader = shader.fragmentShader.replace('#include <emissivemap_fragment>', '');
                material.userData.shader = shader;
            };

            modelScene = gltf.scene;

            // --- WEAPON MAP: mesh uuid → {num, def, role} ---
            // Reads weapon_summary from root node extras and weapon_roles from piece nodes.
            const weaponSummary = {};
            modelScene.traverse(node => {
                if (node.userData && node.userData.weapon_summary) {
                    Object.assign(weaponSummary, node.userData.weapon_summary);
                }
            });
            // Build mesh → weapon info lookup
            const meshWeaponMap = new Map();
            modelScene.traverse(node => {
                if (!node.isMesh) return;
                const weapons = node.userData && node.userData.weapons;
                const roles = node.userData && node.userData.weapon_roles;
                if (!weapons || !weapons.length) return;
                const wnum = weapons[0];
                const summary = weaponSummary[String(wnum)];
                meshWeaponMap.set(node.uuid, {
                    num: wnum,
                    def: summary ? summary.def : null,
                    role: roles ? roles[0] : null,
                });
            });

            // --- DOM WEAPON NAME LOOKUP ---
            // wpn-cat-link uses dashes (e.g. "legkark-heat-ray"),
            // GLB def uses underscores (e.g. "heat_ray") → normalise both to dashes for matching.
            const domWeaponNames = new Map(); // normalised slug → display name
            document.querySelectorAll('[wpn-cat-link]').forEach(el => {
                const slug = el.getAttribute('wpn-cat-link'); // e.g. "legkark-heat-ray"
                const name = el.textContent.trim();           // e.g. "Heat Ray"
                if (slug && name) domWeaponNames.set(slug, name);
            });
            function getWeaponLabel(winfo) {
                if (winfo.def) {
                    const slug = `${unitName}-${winfo.def.replace(/_/g, '-')}`;
                    if (domWeaponNames.has(slug)) return domWeaponNames.get(slug);
                    return slug; // fallback to slug if not found in DOM
                }
                return `Weapon ${winfo.num}`;
            }

            // --- TOOLTIP ELEMENT ---
            const tooltip = document.createElement('div');
            tooltip.style.cssText = [
                'position:fixed','pointer-events:none','display:none',
                'background:rgba(0,0,0,0.78)','color:#fff',
                'font-size:13px','font-family:sans-serif','font-weight:500',
                'padding:5px 10px','border-radius:5px',
                'border:1px solid rgba(255,255,255,0.18)',
                'white-space:nowrap','z-index:9999',
            ].join(';');
            document.body.appendChild(tooltip);

            const raycaster = new THREE.Raycaster();
            const pointer = new THREE.Vector2();
            let tooltipHideTimer = null;

            function showTooltip(x, y, winfo) {
                tooltip.textContent = getWeaponLabel(winfo);
                tooltip.style.display = 'block';
                tooltip.style.left = (x + 14) + 'px';
                tooltip.style.top  = (y - 8) + 'px';
            }
            function hideTooltip() {
                tooltip.style.display = 'none';
            }

            modelScene.traverse(child => {
                if (child.isMesh) {
                    child.material = material;
                    if (shouldHideMesh(child.name)) child.visible = false;
                    else { child.castShadow = true; child.receiveShadow = true; }
                }
            });

            const box = new THREE.Box3().setFromObject(modelScene);
            const center = box.getCenter(new THREE.Vector3());
            modelScene.position.sub(center);
            scene.add(modelScene);

            const boundingSphere = new THREE.Sphere();
            box.getBoundingSphere(boundingSphere);
            const objectRadius = boundingSphere.radius;

            const fovVertical = camera.fov * (Math.PI / 180);
            let distance = Math.abs(objectRadius / Math.sin(fovVertical / 2));
            const aspect = container.offsetWidth / container.offsetHeight;
            if (aspect < 1) { distance = distance / aspect; }
            const finalDistance = distance * VIEWER_CONFIG.initialZoomMultiplier;

            camera.position.set(finalDistance, finalDistance * VIEWER_CONFIG.cameraHeightFactor, finalDistance);
            controls.target.set(0, 0, 0);
            controls.update();

            const maxDim = objectRadius * 2;

            const sCamSize = maxDim * 3.0;
            directionalLight.shadow.camera.left = directionalLight.shadow.camera.bottom = -sCamSize;
            directionalLight.shadow.camera.right = directionalLight.shadow.camera.top = sCamSize;
            directionalLight.shadow.camera.updateProjectionMatrix();

            const groundMat = new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.4 });
            groundMesh = new THREE.Mesh(new THREE.PlaneGeometry(maxDim*10, maxDim*10), groundMat);
            groundMesh.rotation.x = -Math.PI / 2;
            groundMesh.position.y = box.min.y - center.y - 0.01;
            groundMesh.receiveShadow = true;
            scene.add(groundMesh);

            if (gltf.animations && gltf.animations.length) {
                mixer = new THREE.AnimationMixer(modelScene);
                const clip = gltf.animations.find(c => c.name === "walk") || gltf.animations[0];
                currentAction = mixer.clipAction(clip);
                currentAction.play();
                if (animToggleButton) animToggleButton.style.display = 'flex';
            } else if (animToggleButton) animToggleButton.style.display = 'none';

            const renderTarget = new THREE.WebGLRenderTarget(container.offsetWidth, container.offsetHeight, {
                minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat,
                alpha: true, type: THREE.HalfFloatType, samples: isMobile ? 0 : 4
            });

            composer = new EffectComposer(renderer, renderTarget);
            const renderPass = new RenderPass(scene, camera);
            renderPass.clearColor = new THREE.Color(0, 0, 0); renderPass.clearAlpha = 0;
            composer.addPass(renderPass);

            if (performanceSettings.ssaoEnabled) {
                ssaoPass = new SSAOPass(scene, camera, container.offsetWidth, container.offsetHeight);
                ssaoPass.kernelRadius = performanceSettings.ssaoKernelRadius; ssaoPass.kernelSize = performanceSettings.ssaoKernelSize;
                ssaoPass.minDistance = VIEWER_CONFIG.ssaoMinDist; ssaoPass.maxDistance = VIEWER_CONFIG.ssaoMaxDist;
                composer.addPass(ssaoPass);
            }
            smaaPass = new SMAAPass( container.offsetWidth * renderer.getPixelRatio(), container.offsetHeight * renderer.getPixelRatio() );
            composer.addPass(smaaPass);
            outputPass = new OutputPass();
            composer.addPass(outputPass);

            if (isEditorMode) {
                const gui = new GUI({ title: 'Unit Lab' });
                const fGen = gui.addFolder('General');
                fGen.add(VIEWER_CONFIG, 'mouseRotateSpeed', 0.1, 2.0).onChange(v => controls.rotateSpeed = v);
                fGen.add(VIEWER_CONFIG, 'autoRotationSpeed', 0, 2.0);
                fGen.addColor(VIEWER_CONFIG, 'teamColor').onChange(v => { if (material.userData.shader) material.userData.shader.uniforms.teamColor.value.set(v); });
                fGen.add(VIEWER_CONFIG, 'hdrUrl').onFinishChange(v => loadHDR(v));
                fGen.add(VIEWER_CONFIG, 'envMapIntensity', 0, 5).onChange(v => material.envMapIntensity = v);
                fGen.add(VIEWER_CONFIG, 'toneMappingExposure', 0.1, 4.0).onChange(v => renderer.toneMappingExposure = v);

                const fLights = gui.addFolder('Lights Setup');
                const fKey = fLights.addFolder('Key Light (Shadows)');
                fKey.addColor(VIEWER_CONFIG, 'keyLightColor').onChange(v => directionalLight.color.set(v));
                fKey.add(VIEWER_CONFIG, 'keyLightIntensity', 0, 5).onChange(v => directionalLight.intensity = v);
                fKey.add(VIEWER_CONFIG, 'keyLightX', -10, 10).onChange(v => directionalLight.position.x = v);
                fKey.add(VIEWER_CONFIG, 'keyLightY', -10, 10).onChange(v => directionalLight.position.y = v);
                fKey.add(VIEWER_CONFIG, 'keyLightZ', -10, 10).onChange(v => directionalLight.position.z = v);
                fKey.add(VIEWER_CONFIG, 'shadowRadius', 0, 25).onChange(v => directionalLight.shadow.radius = v);
                fKey.add(VIEWER_CONFIG, 'shadowBlurSamples', 0, 25).step(1).onChange(v => directionalLight.shadow.blurSamples = v);

                const fFill = fLights.addFolder('Fill Light');
                fFill.addColor(VIEWER_CONFIG, 'fillLightColor').onChange(v => directionalLight2.color.set(v));
                fFill.add(VIEWER_CONFIG, 'fillLightIntensity', 0, 5).onChange(v => directionalLight2.intensity = v);
                fFill.add(VIEWER_CONFIG, 'fillLightX', -10, 10).onChange(v => directionalLight2.position.x = v);
                fFill.add(VIEWER_CONFIG, 'fillLightY', -10, 10).onChange(v => directionalLight2.position.y = v);
                fFill.add(VIEWER_CONFIG, 'fillLightZ', -10, 10).onChange(v => directionalLight2.position.z = v);

                const fAmb = fLights.addFolder('Ambient Light');
                fAmb.addColor(VIEWER_CONFIG, 'ambientLightColor').onChange(v => ambientLight.color.set(v));
                fAmb.add(VIEWER_CONFIG, 'ambientLightIntensity', 0, 2).onChange(v => ambientLight.intensity = v);

                const fPost = gui.addFolder('Post Processing');
                fPost.add(VIEWER_CONFIG, 'ssaoEnabled').onChange(v => ssaoPass.enabled = v);
                fPost.add(VIEWER_CONFIG, 'ssaoRadius', 0, 64).onChange(v => ssaoPass.kernelRadius = v);
                fPost.add(VIEWER_CONFIG, 'ssaoMinDist', 0.001, 0.1).onChange(v => ssaoPass.minDistance = v);
                fPost.add(VIEWER_CONFIG, 'ssaoMaxDist', 0.1, 10).onChange(v => ssaoPass.maxDistance = v);

                const fExtra = gui.addFolder('Extras');
                fExtra.add(VIEWER_CONFIG, 'roughnessOffset', -1, 1).onChange(v => { if(material.userData.shader) material.userData.shader.uniforms.roughnessOffset.value = v; });
                fExtra.add({ toggleShadow: () => { areShadowsVisible = !areShadowsVisible; if(groundMesh) groundMesh.visible = areShadowsVisible; }}, 'toggleShadow');
            }

            function exportHighResPNG(w, h) {
                const origW = container.offsetWidth, origH = container.offsetHeight;
                const origPR = renderer.getPixelRatio();
                const origAsp = camera.aspect;

                requestAnimationFrame(() => {
                    renderer.setPixelRatio(1); renderer.setSize(w, h, false);
                    composer.setPixelRatio(1); composer.setSize(w, h);
                    camera.aspect = w / h; camera.updateProjectionMatrix();

                    if(ssaoPass) ssaoPass.setSize(w, h);
                    if(smaaPass) smaaPass.setSize(w, h);

                    renderer.setClearColor(0x000000, 0);
                    composer.render();

                    try {
                        const dataURL = renderer.domElement.toDataURL('image/png');
                        const link = document.createElement('a');
                        link.download = `${unitName}_render.png`;
                        link.href = dataURL;
                        document.body.appendChild(link); link.click(); document.body.removeChild(link);
                    } catch(e) { console.error("Export failed:", e); }

                    renderer.setPixelRatio(origPR); renderer.setSize(origW, origH, false);
                    composer.setPixelRatio(origPR); composer.setSize(origW, origH);
                    camera.aspect = origAsp; camera.updateProjectionMatrix();

                    if(ssaoPass) ssaoPass.setSize(origW, origH);
                    if(smaaPass) smaaPass.setSize(origW * origPR, origH * origPR);
                    composer.render();
                });
            }
            if (exportButton) exportButton.addEventListener('click', () => {
                const aspect = container.offsetWidth / container.offsetHeight;
                exportHighResPNG(VIEWER_CONFIG.exportWidth, Math.round(VIEWER_CONFIG.exportWidth / aspect));
            });

            window.addEventListener('keydown', (e) => {
                if (e.key === 'Alt') { e.preventDefault(); isAltDown = true; container.style.cursor = 'move'; controls.enabled = false; }
                if (e.code === 'Space' && mixer && currentAction && !e.repeat) currentAction.paused = !currentAction.paused;
            });
            window.addEventListener('keyup', (e) => { if (e.key === 'Alt') { isAltDown = false; container.style.cursor = 'grab'; controls.enabled = true; } });
            window.addEventListener('mousemove', (e) => {
                if (isAltDown) { lightGroup.rotation.y += e.movementX * 0.01; hideTooltip(); return; }
                if (meshWeaponMap.size === 0) return;
                const rect = renderer.domElement.getBoundingClientRect();
                if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
                    hideTooltip(); return;
                }
                pointer.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
                pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
                raycaster.setFromCamera(pointer, camera);
                const hits = raycaster.intersectObject(modelScene, true);
                const hit = hits.find(h => meshWeaponMap.has(h.object.uuid));
                if (hit) {
                    showTooltip(e.clientX, e.clientY, meshWeaponMap.get(hit.object.uuid));
                } else {
                    hideTooltip();
                }
            });
            renderer.domElement.addEventListener('touchstart', (e) => {
                if (meshWeaponMap.size === 0 || e.touches.length !== 1) return;
                const touch = e.touches[0];
                const rect = renderer.domElement.getBoundingClientRect();
                pointer.x = ((touch.clientX - rect.left) / rect.width)  * 2 - 1;
                pointer.y = -((touch.clientY - rect.top)  / rect.height) * 2 + 1;
                raycaster.setFromCamera(pointer, camera);
                const hits = raycaster.intersectObject(modelScene, true);
                const hit = hits.find(h => meshWeaponMap.has(h.object.uuid));
                if (hit) {
                    showTooltip(touch.clientX, touch.clientY, meshWeaponMap.get(hit.object.uuid));
                    clearTimeout(tooltipHideTimer);
                    tooltipHideTimer = setTimeout(hideTooltip, 2000);
                }
            }, { passive: true });

            controls.addEventListener('start', () => { isInteracting = true; turntableActive = false; clearTimeout(idleTimer); });
            controls.addEventListener('end', () => { isInteracting = false; if (isTurntableEnabledByUser) idleTimer = setTimeout(() => { if (!isInteracting) turntableActive = true; }, 3000); });
            if (isTurntableEnabledByUser) turntableActive = true;

            if (animToggleButton) animToggleButton.addEventListener('click', () => { if(mixer && currentAction) currentAction.paused = !currentAction.paused; });
            if (rotateToggleButton) rotateToggleButton.addEventListener('click', () => { isTurntableEnabledByUser = !isTurntableEnabledByUser; turntableActive = isTurntableEnabledByUser; if(!turntableActive) clearTimeout(idleTimer); });
            if (shadowToggleButton) shadowToggleButton.addEventListener('click', () => { areShadowsVisible = !areShadowsVisible; if(groundMesh) groundMesh.visible = areShadowsVisible; });

            const envRotMat4 = new THREE.Matrix4();
            const envRotMat3 = new THREE.Matrix3();

            // --- FPS LIMITER: MAX 120 ---
            const fpsLimit = 120;
            const interval = 1000 / fpsLimit;
            let lastTime = 0;

            const animate = (timestamp) => {
                requestAnimationFrame(animate);

                const delta = timestamp - lastTime;

                if (delta > interval) {
                    lastTime = timestamp - (delta % interval);

                    const clockDelta = clock.getDelta();
                    timeUniform.value = clock.elapsedTime;

                    if (material.userData.shader && material.userData.shader.uniforms.envMapRotation) {
                        envRotMat4.makeRotationY(lightGroup.rotation.y);
                        envRotMat3.setFromMatrix4(envRotMat4);
                        material.userData.shader.uniforms.envMapRotation.value.copy(envRotMat3);
                    }

                    if (turntableActive && !isInteracting && !isAltDown) modelScene.rotation.y += VIEWER_CONFIG.autoRotationSpeed * clockDelta;
                    if (mixer) mixer.update(clockDelta);
                    controls.update();
                    composer.render();
                }
            };

            const handleResize = () => {
                const w = container.offsetWidth; const h = container.offsetHeight;
                if (w === 0 || h === 0) return;
                camera.aspect = w / h; camera.updateProjectionMatrix();
                renderer.setSize(w, h); composer.setSize(w, h);
                if(ssaoPass) ssaoPass.setSize(w, h);
                if(smaaPass) smaaPass.setSize(w * renderer.getPixelRatio(), h * renderer.getPixelRatio());
                composer.render();
            };
            const resizeObserver = new ResizeObserver(handleResize);
            resizeObserver.observe(container);

            lastTime = performance.now();
            requestAnimationFrame(animate);

        }).catch(err => {
            console.error("LOAD ERROR:", err);
            if(noticeElement) { noticeElement.textContent = "Error loading model."; noticeElement.style.display = 'block'; }
        });
    }, 900);
});
