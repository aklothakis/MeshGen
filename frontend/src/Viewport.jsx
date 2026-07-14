import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// Renders the geometry surface and/or the generated mesh in a Three.js scene.
export default function Viewport({ geometry, mesh, showMode, meshLayers }) {
  const mountRef = useRef(null)
  const stateRef = useRef({})

  // One-time scene setup.
  useEffect(() => {
    const mount = mountRef.current
    const width = mount.clientWidth
    const height = mount.clientHeight

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#0a0e17')

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 5000)
    camera.position.set(4, 3, 6)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08

    scene.add(new THREE.HemisphereLight('#9fc6ff', '#20242e', 1.1))
    const key = new THREE.DirectionalLight('#ffffff', 1.4)
    key.position.set(5, 8, 6)
    scene.add(key)
    const rim = new THREE.DirectionalLight('#6ca8ff', 0.6)
    rim.position.set(-6, -2, -4)
    scene.add(rim)

    const grid = new THREE.GridHelper(20, 40, '#1b2740', '#141b2b')
    grid.position.y = -0.001
    scene.add(grid)

    const root = new THREE.Group()
    scene.add(root)

    stateRef.current = { scene, camera, renderer, controls, root, mount }

    let raf
    const animate = () => {
      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(animate)
    }
    animate()

    const onResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [])

  // Rebuild displayed objects when data or mode changes.
  useEffect(() => {
    const st = stateRef.current
    if (!st.root) return
    const { root, camera, controls } = st
    while (root.children.length) {
      const c = root.children.pop()
      c.geometry?.dispose?.()
      c.material?.dispose?.()
    }

    let bbox = null
    const wantMesh = showMode === 'mesh' && mesh
    const wantGeom = showMode === 'geometry' || !wantMesh

    if (wantGeom && geometry) {
      bbox = addGeometry(root, geometry)
    }
    if (wantMesh) {
      bbox = addMesh(root, mesh, meshLayers)
    }

    // Only reframe the camera when the data/mode changes, not on layer toggles.
    const key = `${showMode}|${geometry ? 'g' : ''}|${mesh ? 'm' : ''}`
    if (bbox && st.lastFrameKey !== key) {
      frameCamera(camera, controls, bbox)
      st.lastFrameKey = key
    }
  }, [geometry, mesh, showMode, meshLayers])

  return <div className="viewport" ref={mountRef} />
}

function toBufferGeometry(positions, indices) {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  if (indices) g.setIndex(indices)
  g.computeVertexNormals()
  return g
}

function addGeometry(root, geometry) {
  const surf = toBufferGeometry(geometry.positions, geometry.indices)
  const mat = new THREE.MeshStandardMaterial({
    color: '#3f6fd6', metalness: 0.35, roughness: 0.45, side: THREE.DoubleSide,
  })
  root.add(new THREE.Mesh(surf, mat))

  if (geometry.lines?.length) {
    const lg = new THREE.BufferGeometry()
    lg.setAttribute('position', new THREE.Float32BufferAttribute(geometry.positions, 3))
    lg.setIndex(geometry.lines)
    root.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
      color: '#9db8ee', transparent: true, opacity: 0.35,
    })))
  }
  if (geometry.leading_edge?.length) {
    const le = new THREE.BufferGeometry()
    le.setAttribute('position', new THREE.Float32BufferAttribute(geometry.leading_edge, 3))
    root.add(new THREE.Line(le, new THREE.LineBasicMaterial({ color: '#ffcf5c', linewidth: 2 })))
  }
  return { min: geometry.bbox_min, max: geometry.bbox_max }
}

function lineSet(positions, index, color, opacity, depthWrite = true) {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  g.setIndex(index)
  const m = new THREE.LineBasicMaterial({
    color, transparent: opacity < 1, opacity, depthWrite,
  })
  return new THREE.LineSegments(g, m)
}

function addMesh(root, mesh, layers) {
  const L = layers || { wall: true, cut: true, farfield: true }

  // Solid body -- opaque so it reads clearly against the wireframe.
  const wall = toBufferGeometry(mesh.positions, mesh.wall_indices)
  const wallMesh = new THREE.Mesh(wall, new THREE.MeshStandardMaterial({
    color: '#3a5a9a', metalness: 0.3, roughness: 0.5, side: THREE.DoubleSide,
  }))
  root.add(wallMesh)
  if (mesh.wall_lines?.length) {
    root.add(lineSet(mesh.positions, mesh.wall_lines, '#bfe0ff', 0.55))
  }

  // Outer domain boundary -- dim, so the domain extent is outlined but recessed.
  if (L.farfield && mesh.farfield_lines?.length) {
    root.add(lineSet(mesh.positions, mesh.farfield_lines, '#54617e', 0.5, false))
  }
  // Radial cut planes -- bright, reveal wall->farfield clustering (BL bunching).
  if (L.cut && mesh.cut_lines?.length) {
    root.add(lineSet(mesh.positions, mesh.cut_lines, '#37d0c0', 0.55, false))
  }
  return { min: mesh.bbox_min, max: mesh.bbox_max }
}

function frameCamera(camera, controls, bbox) {
  const min = new THREE.Vector3(...bbox.min)
  const max = new THREE.Vector3(...bbox.max)
  const center = min.clone().add(max).multiplyScalar(0.5)
  const size = max.clone().sub(min).length() || 1
  controls.target.copy(center)
  // Side-and-above 3/4 view: body length (x) reads across, cross-section visible.
  camera.position.copy(center).add(new THREE.Vector3(1.35, 0.55, 0.95).multiplyScalar(size * 0.8))
  camera.near = size * 0.001
  camera.far = size * 50
  camera.updateProjectionMatrix()
  controls.update()
}
