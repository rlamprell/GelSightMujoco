"""
    Gomes' Gelsight driver

    This takes in a depth map and combines it with a background image (taken from a real
    gelsight sensor) and combines them both with mathematical techniques to produce
    a gelsight output.

    The code is a slightly modified version of what can be found here and is mostly unchanged:
        https://github.com/danfergo/gelsight_simulation
"""


import cv2
import numpy                    as np
import scipy.ndimage.filters    as fi


def show_normalized_img(name, img):
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)
    return draw


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def gaus_noise(image, sigma):
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def derivative(mat, direction):
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0


def tangent(mat):
    dx = derivative(mat, 'x')
    dy = derivative(mat, 'y')
    img_shape = np.shape(mat)
    _1 = np.repeat([1.0], img_shape[0] * img_shape[1]).reshape(img_shape).astype(dx.dtype)
    unormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unormalized, axis=2)
    return (unormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2))


def solid_color_img(color, size):
    image = np.zeros(size + (3,), np.float64)
    image[:] = color
    return image


def add_overlay(rgb, alpha, color):
    s = np.shape(alpha)

    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))  # * 10.0

    overlay = solid_color_img(color, s)

    foreground = opacity3 * overlay
    background = (1.0 - opacity3) * rgb.astype(np.float64)
    res = background + foreground

    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)

    return res


# GelSight Simulation
class SimulationApproach:
    def __init__(self, **config):
        self.light_sources = config['light_sources']
        self.background = config['background_img']
        self.px2m_ratio = config['px2m_ratio']
        self.elastomer_thickness = config['elastomer_thickness']
        self.min_depth = config['min_depth']

        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 5

        self.ka = config['ka'] or 0.8

        self.texture_sigma = config['texture_sigma'] or 0.00001
        self.t = config['t'] if 't' in config else 3
        self.sigma = config['sigma'] if 'sigma' in config else 7
        self.kernel_size = config['sigma'] if 'sigma' in config else 21

        self.max_depth = self.min_depth + self.elastomer_thickness


    def protrusion_map(self, original, not_in_touch):
        protrusion_map = np.copy(original)
        protrusion_map[not_in_touch >= self.max_depth] = self.max_depth
        return protrusion_map


    def segments(self, depth_map):
        
        not_in_touch = np.copy(depth_map)
        not_in_touch[not_in_touch < self.max_depth] = 0.0
        not_in_touch[not_in_touch >= self.max_depth] = 1.0
        in_touch = 1 - not_in_touch

        return not_in_touch, in_touch


    def internal_shadow(self, elastomer_depth):
        elastomer_depth_inv = self.max_depth - elastomer_depth
        elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, self.elastomer_thickness), (0.0, 1.0))
        return elastomer_depth_inv


    def apply_elastic_deformation_v1(self, protrusion_depth, not_in_touch, in_touch):
        kernel = gkern2(15, 7)
        deformation = self.max_depth - protrusion_depth

        for i in range(5):
            deformation = cv2.filter2D(deformation, -1, kernel)

        return 30 * -deformation * not_in_touch + (protrusion_depth * in_touch)


    def apply_elastic_deformation(self, protrusion_depth, not_in_touch, in_touch):
        protrusion_depth = - (protrusion_depth - self.max_depth)

        kernel = gkern2(self.kernel_size, self.sigma)
        deformation = protrusion_depth

        deformation2 = protrusion_depth
        kernel2 = gkern2(52, 9)

        for i in range(self.t):
            deformation_ = cv2.filter2D(deformation, -1, kernel)
            r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
            deformation = np.maximum(r * deformation_, protrusion_depth)

            deformation2_ = cv2.filter2D(deformation2, -1, kernel2)
            r = np.max(protrusion_depth) / np.max(deformation2_) if np.max(deformation2_) > 0 else 1
            deformation2 = np.maximum(r * deformation2_, protrusion_depth)

        deformation_v1 = self.apply_elastic_deformation_v1(protrusion_depth, not_in_touch, in_touch)

        for i in range(self.t):
            deformation_ = cv2.filter2D(deformation2, -1, kernel)
            r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
            deformation2 = np.maximum(r * deformation_, protrusion_depth)


        deformation_x = 2 * deformation - deformation2
      

        return self.max_depth - deformation_x


    # use phong illuminations to 
    def phong_illumination(self, T, source_dir, kd, ks, alpha):
        dot = np.dot(T, np.array(source_dir)).astype(np.float64)
        difuse_l = dot * kd
        difuse_l[difuse_l < 0] = 0.0

        dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)

        R = 2.0 * dot3 * T - source_dir
        V = [0.0, 0.0, 1.0]

        spec_l = np.power(np.dot(R, V), alpha) * ks

        return difuse_l + spec_l


    # main worker class
    # -- call and pass depth map in order to create a simulated gelsight image
    def generate(self, obj_depth, return_depth=False):
        
        not_in_touch, in_touch      = self.segments(obj_depth)
        protrusion_depth            = self.protrusion_map(obj_depth, not_in_touch)
        elastomer_depth             = self.apply_elastic_deformation(protrusion_depth, not_in_touch, in_touch)

        textured_elastomer_depth    = gaus_noise(elastomer_depth, self.texture_sigma)


        out = self.ka * self.background
        out = add_overlay(out, self.internal_shadow(protrusion_depth), (0.0, 0.0, 0.0))

        T = tangent(textured_elastomer_depth / self.px2m_ratio)

        for light in self.light_sources:
            ks      = light['ks'] if 'ks' in light else self.default_ks
            kd      = light['kd'] if 'kd' in light else self.default_kd
            alpha   = light['alpha'] if 'alpha' in light else self.default_alpha
            out     = add_overlay(out, self.phong_illumination(T, light['position'], kd, ks, alpha), light['color'])

        kernel = gkern2(3, 1)
        out = cv2.filter2D(out, -1, kernel)


        if return_depth:
            return out, elastomer_depth
        return out