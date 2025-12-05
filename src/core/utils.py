import trimesh


def Ball(scale=(1.0, 1.0, 1.0)):
    center = [0, 0, 0]
    radius = 0.15 * min(scale)
    return trimesh.creation.icosphere(radius=radius, center=center, subdivisions=4)


def Box(scale=(1.0, 1.0, 1.0)):
    center = [0, 0, 0]
    extents = [0.3 * s for s in scale]
    return trimesh.creation.box(
        extents=extents, transform=trimesh.transformations.translation_matrix(center)
    )


def Cylinder(scale=(1.0, 1.0, 1.0)):
    radius = 0.15 * min(scale[0], scale[1])
    height = 0.3 * scale[2]
    center = [0, 0, 0]
    return trimesh.creation.cylinder(
        radius=radius, height=height, center=center, resolution=100
    )

def Sphere(scale=(1.0, 1.0, 1.0)):
    radius = 0.15 * min(scale)
    center = [0, 0, 0]
    return trimesh.creation.uv_sphere(radius=radius, center=center, resolution=100)


def Cone(scale=(1.0, 1.0, 1.0)):
    radius = 0.15 * min(scale[0], scale[1])
    height = 0.3 * scale[2]
    center = [0, 0, 0]
    return trimesh.creation.cone(
        radius=radius, height=height, center=center, resolution=100
    )


def Torus(scale=(1.0, 1.0, 1.0)):
    base = 0.15 * min(scale)
    outer_radius = base
    inner_radius = base * 0.5
    center = [0, 0, 0]
    return trimesh.creation.torus(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        center=center,
        resolution=100,
    )


GEOMS = {
    "Ball": Ball,
    "Box": Box,
    "Cylinder": Cylinder,
    "Sphere": Sphere,
    "Cone": Cone,
    "Torus": Torus,
}
