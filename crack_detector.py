"""
AquaInspect — Classical Image Processing-Based Defect Detection
Member 2: Crack Detection, Skeletonization, and Quantitative Measurement Module

Author: Member 2 (Crack Detection Developer)
Technologies: Python, OpenCV (cv2), NumPy, scikit-image, Matplotlib
"""

import os
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class CrackDetector:
    """
    Classical Digital Image Processing pipeline for detecting structural cracks
    in underwater pipeline and bridge pier inspection imagery.
    """

    def __init__(
        self,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        blackhat_kernel_size: int = 15,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
        min_contour_area: int = 80,
        min_aspect_ratio: float = 1.8,
        min_length_thresh: int = 50,
        pixel_to_mm_ratio: Optional[float] = None
    ):
        """
        Initialize the crack detector with configurable tuning parameters.

        Args:
            bilateral_d: Diameter of each pixel neighborhood for bilateral filtering.
            bilateral_sigma_color: Filter sigma in the color space.
            bilateral_sigma_space: Filter sigma in the coordinate space.
            blackhat_kernel_size: Structuring element size for the Black-Hat morphological filter.
            canny_thresh1: First threshold for Canny edge detector hysteresis.
            canny_thresh2: Second threshold for Canny edge detector hysteresis.
            min_contour_area: Minimum pixel area to discard small noise specks (grit/sand).
            min_aspect_ratio: Minimum elongation ratio (length/width) for small features.
            min_length_thresh: Minimum absolute length to classify as an extended crack.
            pixel_to_mm_ratio: Conversion factor (mm per pixel) if camera calibration is known.
        """
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.blackhat_kernel_size = blackhat_kernel_size
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.min_contour_area = min_contour_area
        self.min_aspect_ratio = min_aspect_ratio
        self.min_length_thresh = min_length_thresh
        self.pixel_to_mm_ratio = pixel_to_mm_ratio

    def load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load an image from a filepath or validate an in-memory numpy array.
        Returns image in BGR format.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or numpy.ndarray")

    def preprocess(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1 & 2: Grayscale conversion and edge-preserving bilateral filtering.

        Returns:
            gray: Standard grayscale image.
            filtered: Bilateral filtered grayscale image (smooths texture, keeps edges).
        """
        if len(bgr_image.shape) == 3:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_image.copy()

        # Edge-preserving Bilateral Filter
        filtered = cv2.bilateralFilter(
            gray,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        return gray, filtered

    def segment_cracks(
        self,
        filtered_gray: np.ndarray,
        method: str = "hybrid"
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Step 3, 4 & 5: Segment cracks using classical techniques, morphology, and filtering.

        Args:
            filtered_gray: Preprocessed grayscale image.
            method: 'hybrid' (recommended, BlackHat + Adaptive + Canny refinement),
                    'blackhat' (BlackHat morphological transform + Otsu),
                    'canny' (pure Canny edge detection + morphology).

        Returns:
            clean_mask: Cleaned binary crack mask (255 = crack, 0 = background).
            intermediates: Dictionary of intermediate response matrices for analysis.
        """
        intermediates = {}

        # 1. Morphological Black-Hat Transform:
        # Isolates dark narrow features (cracks) that are darker than local surroundings
        k_size = self.blackhat_kernel_size
        k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        blackhat = cv2.morphologyEx(filtered_gray, cv2.MORPH_BLACKHAT, k_rect)
        intermediates["blackhat"] = blackhat

        # 2. Canny Edge Detection
        canny_edges = cv2.Canny(
            filtered_gray,
            self.canny_thresh1,
            self.canny_thresh2
        )
        intermediates["canny"] = canny_edges

        # 3. Adaptive Thresholding on Black-Hat response
        adaptive_thresh = cv2.adaptiveThreshold(
            blackhat,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=-2
        )
        intermediates["adaptive_thresh"] = adaptive_thresh

        # Select initial binary segmentation
        if method == "canny":
            initial_binary = canny_edges
        elif method == "blackhat":
            _, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            initial_binary = otsu
            intermediates["otsu"] = otsu
        else:  # hybrid
            # Combines the strong intensity contrast of adaptive black-hat
            # with edge alignment
            initial_binary = adaptive_thresh

        # 4. Morphological Closing: Bridges fragmented crack lines
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(initial_binary, cv2.MORPH_CLOSE, close_kernel)
        intermediates["closed"] = closed

        # 5. Connected Component Filtering: Remove tiny specks / keep elongated crack structures
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        clean_mask = np.zeros_like(closed, dtype=np.uint8)

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            w = stats[label_idx, cv2.CC_STAT_WIDTH]
            h = stats[label_idx, cv2.CC_STAT_HEIGHT]
            length_est = max(w, h)
            min_dim = max(min(w, h), 1)
            aspect_ratio = length_est / min_dim

            # Retain components that are either significantly elongated or have substantive area
            is_valid_crack = (
                (area >= self.min_contour_area and aspect_ratio >= self.min_aspect_ratio) or
                (length_est >= self.min_length_thresh and area >= self.min_contour_area // 2) or
                (area >= self.min_contour_area * 3)
            )

            if is_valid_crack:
                clean_mask[labels == label_idx] = 255

        intermediates["clean_mask"] = clean_mask
        return clean_mask, intermediates

    def compute_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Step 6: Skeletonization (Thinning).
        Reduces binary crack masks down to a single-pixel wide topological centerline.
        """
        if np.sum(binary_mask) == 0:
            return np.zeros_like(binary_mask, dtype=np.uint8)

        # skimage skeletonize takes boolean array
        bool_skeleton = skeletonize(binary_mask > 0)
        skeleton_uint8 = (bool_skeleton.astype(np.uint8)) * 255
        return skeleton_uint8

    def measure_cracks(
        self,
        binary_mask: np.ndarray,
        skeleton: np.ndarray
    ) -> Dict[str, Any]:
        """
        Step 7: Quantitative measurements of cracks.
        Calculates Crack Area, Skeleton Length, and local Width (via Distance Transform).

        Returns:
            Dictionary with both pixel and physical (if calibrated) metrics.
        """
        crack_detected = bool(np.any(binary_mask > 0))

        if not crack_detected:
            return {
                "crack_detected": False,
                "crack_area_px": 0,
                "crack_length_px": 0.0,
                "mean_width_px": 0.0,
                "max_width_px": 0.0,
                "num_crack_segments": 0,
                "crack_area_mm2": 0.0 if self.pixel_to_mm_ratio else None,
                "crack_length_mm": 0.0 if self.pixel_to_mm_ratio else None,
                "mean_width_mm": 0.0 if self.pixel_to_mm_ratio else None,
                "max_width_mm": 0.0 if self.pixel_to_mm_ratio else None,
            }

        # 1. Total Crack Area (pixel count)
        area_px = int(np.sum(binary_mask > 0))

        # 2. Number of discrete crack components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        num_segments = int(num_labels - 1)  # subtract background

        # 3. Crack Length from Skeleton
        # Basic length is sum of skeleton pixels.
        # For higher accuracy, count diagonal transitions vs orthogonal transitions
        skeleton_pts = skeleton > 0
        length_px = float(np.sum(skeleton_pts))

        # 4. Crack Width using Euclidean Distance Transform
        # At each skeleton pixel, dist_transform value = radius of inscribed circle.
        # Total local width = 2 * radius.
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        if length_px > 0:
            skeleton_radii = dist_transform[skeleton_pts]
            local_widths = skeleton_radii * 2.0
            mean_width_px = float(np.mean(local_widths))
            max_width_px = float(np.max(local_widths))
        else:
            # Fallback estimation
            mean_width_px = float(area_px / max(length_px, 1.0))
            max_width_px = mean_width_px

        metrics = {
            "crack_detected": True,
            "crack_area_px": area_px,
            "crack_length_px": round(length_px, 2),
            "mean_width_px": round(mean_width_px, 2),
            "max_width_px": round(max_width_px, 2),
            "num_crack_segments": num_segments
        }

        # Physical units conversion if calibration ratio is available
        if self.pixel_to_mm_ratio is not None and self.pixel_to_mm_ratio > 0:
            scale = self.pixel_to_mm_ratio
            metrics["crack_area_mm2"] = round(area_px * (scale ** 2), 2)
            metrics["crack_length_mm"] = round(length_px * scale, 2)
            metrics["mean_width_mm"] = round(mean_width_px * scale, 2)
            metrics["max_width_mm"] = round(max_width_px * scale, 2)
        else:
            metrics["crack_area_mm2"] = None
            metrics["crack_length_mm"] = None
            metrics["mean_width_mm"] = None
            metrics["max_width_mm"] = None

        return metrics

    def create_visual_overlay(
        self,
        bgr_image: np.ndarray,
        binary_mask: np.ndarray,
        skeleton: np.ndarray,
        mask_color: Tuple[int, int, int] = (0, 0, 255),       # Vibrant Red (BGR)
        skeleton_color: Tuple[int, int, int] = (0, 255, 255), # Bright Yellow (BGR)
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Generate a diagnostic visual overlay with crack masks and skeleton spines.
        """
        overlay = bgr_image.copy()

        # Color the crack region
        crack_indices = binary_mask > 0
        if np.any(crack_indices):
            colored_mask = np.zeros_like(bgr_image)
            colored_mask[crack_indices] = mask_color
            cv2.addWeighted(colored_mask, alpha, overlay, 1.0, 0, overlay)

        # Draw skeleton spine (1-pixel or dilated for clear visual presentation)
        skeleton_indices = skeleton > 0
        if np.any(skeleton_indices):
            # Dilate skeleton by 1 pixel for better display visibility on high-res images
            dilated_skel = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            overlay[dilated_skel > 0] = skeleton_color

        return overlay

    def generate_inspection_figure(
        self,
        bgr_image: np.ndarray,
        binary_mask: np.ndarray,
        skeleton: np.ndarray,
        overlay: np.ndarray,
        metrics: Dict[str, Any],
        title: str = "AquaInspect — Defect Detection Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Creates a high-resolution 4-panel diagnostic figure for reports and papers.
        """
        rgb_original = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(2, 2, figsize=(14, 11))

        # Panel 1: Input Enhanced Image
        axs[0, 0].imshow(rgb_original)
        axs[0, 0].set_title("(a) Input Enhanced Underwater Image", fontsize=12, fontweight="bold")
        axs[0, 0].axis("off")

        # Panel 2: Binary Crack Mask
        axs[0, 1].imshow(binary_mask, cmap="gray")
        axs[0, 1].set_title("(b) Binary Crack Segmentation Mask", fontsize=12, fontweight="bold")
        axs[0, 1].axis("off")

        # Panel 3: Crack Skeleton
        axs[1, 0].imshow(skeleton, cmap="magma")
        axs[1, 0].set_title("(c) Crack Centerline Skeleton (1-px Spine)", fontsize=12, fontweight="bold")
        axs[1, 0].axis("off")

        # Panel 4: Overlay & Quantitative Legend
        axs[1, 1].imshow(rgb_overlay)
        status_text = (
            f"Crack Detected: {'YES' if metrics['crack_detected'] else 'NO'}\n"
            f"Segments: {metrics['num_crack_segments']}\n"
            f"Length: {metrics['crack_length_px']} px\n"
            f"Mean Width: {metrics['mean_width_px']} px\n"
            f"Max Width: {metrics['max_width_px']} px\n"
            f"Area: {metrics['crack_area_px']} px"
        )
        axs[1, 1].text(
            0.03, 0.97, status_text,
            transform=axs[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.75, edgecolor="yellow")
        )
        axs[1, 1].set_title("(d) Diagnostic Overlay (Red: Mask, Yellow: Spine)", fontsize=12, fontweight="bold")
        axs[1, 1].axis("off")

        fig.suptitle(title, fontsize=16, fontweight="heavy", y=0.98)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

        return fig

    def process(
        self,
        image_input: Union[str, np.ndarray],
        method: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Execute the full Member 2 crack detection and measurement pipeline.

        Returns:
            Dictionary containing:
            - 'original_bgr': Input image in BGR
            - 'gray': Grayscale image
            - 'filtered': Bilateral filtered image
            - 'mask': Binary crack mask (0/255)
            - 'skeleton': 1-pixel centerline skeleton (0/255)
            - 'overlay': BGR diagnostic overlay image
            - 'metrics': Quantitative measurements dict
            - 'intermediates': Intermediate morphological and edge stages
        """
        bgr = self.load_image(image_input)
        gray, filtered = self.preprocess(bgr)
        mask, intermediates = self.segment_cracks(filtered, method=method)
        skeleton = self.compute_skeleton(mask)
        metrics = self.measure_cracks(mask, skeleton)
        overlay = self.create_visual_overlay(bgr, mask, skeleton)

        return {
            "original_bgr": bgr,
            "gray": gray,
            "filtered": filtered,
            "mask": mask,
            "skeleton": skeleton,
            "overlay": overlay,
            "metrics": metrics,
            "intermediates": intermediates
        }


# Convenience Function for Seamless Integration (Member 3 & Member 4)
def detect_and_measure_cracks(
    image_or_path: Union[str, np.ndarray],
    pixel_to_mm_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    Standard top-level interface for Member 3 (severity scoring)
    and Member 4 (Streamlit dashboard integration).

    Usage:
        from modules.crack_detector import detect_and_measure_cracks
        result = detect_and_measure_cracks("enhanced_image.jpg")
        mask = result['mask']
        length = result['metrics']['crack_length_px']
    """
    detector = CrackDetector(pixel_to_mm_ratio=pixel_to_mm_ratio)
    return detector.process(image_or_path)
