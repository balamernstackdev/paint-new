import cv2
import numpy as np
from PIL import Image
import streamlit as st
from scipy import sparse
from app_config.constants import ColorizerConfig

# Try to import adaptive processing, but make it optional
try:
    from .adaptive_processing import (
        get_adaptive_blur_kernel,
        apply_bilateral_blur,
        classify_object,
        get_object_params,
        ObjectType
    )
    ADAPTIVE_AVAILABLE = True
except ImportError as e:
    # Fallback if adaptive module not available
    ADAPTIVE_AVAILABLE = False
    import logging
    logging.warning(f"Adaptive processing not available: {e}")

class ColorTransferEngine:
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert HEX string to RGB tuple.
        
        Args:
            hex_color: Hex color string (e.g., '#FF0000' or 'FF0000')
            
        Returns:
            Tuple[int, int, int]: RGB values (0-255)
            
        Raises:
            ValueError: If hex_color is invalid
        """
        if not isinstance(hex_color, str):
            raise ValueError(f"hex_color must be a string, got {type(hex_color)}")
        
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) != 6:
            raise ValueError(f"hex_color must be  6 characters (got {len(hex_color)}): {hex_color}")
        
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError as e:
            raise ValueError(f"Invalid hex color '{hex_color}': {e}")

    @staticmethod
    def apply_color(image_rgb, mask, target_color_hex, intensity=1.0, seed_point=None, use_adaptive=False):
        """
        Apply color with optional adaptive processing (DISABLED by default).
        
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format
            mask: Boolean or uint8 mask array (H, W)
            target_color_hex: Target color as hex string
            intensity: Blending intensity (0.0-1.0)
            seed_point: Optional (x, y) click point for object classification
            use_adaptive: If True, uses adaptive blur (slower, experimental)
            
        Returns:
            NumPy array: Colored image
            
        Raises:
            ValueError: If inputs are invalid
            
        Note:
            Adaptive processing adds edge detection and object classification,
            which is slower. Default legacy mode is optimized for speed
            and texture preservation.
        """
        # Input validation
        if not isinstance(image_rgb, np.ndarray):
            raise ValueError(f"image_rgb must be numpy array, got {type(image_rgb)}")
        
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must be (H, W, 3), got {image_rgb.shape}")
        
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"mask must be numpy array, got {type(mask)}")
        
        if len(mask.shape) != 2:
            raise ValueError(f"mask must be 2D array, got shape {mask.shape}")
        
        if mask.shape[:2] != image_rgb.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} doesn't match image {image_rgb.shape[:2]}")
        
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be between 0 and 1, got {intensity}")
        
        # Ensure input is standard format
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Adaptive Blur Selection
        mask_float = mask.astype(np.float32)
        
        if use_adaptive and ADAPTIVE_AVAILABLE:
            try:
                # Detect optimal blur kernel based on edge density
                blur_kernel = get_adaptive_blur_kernel(mask, image_rgb)
                
                # Optionally detect if texture preservation needed
                if seed_point is not None:
                    try:
                        obj_type = classify_object(mask, image_rgb, seed_point)
                        params = get_object_params(obj_type)
                        
                        if params['use_bilateral']:
                            # Use bilateral filter for textured surfaces
                            mask_soft = apply_bilateral_blur(mask_float, preserve_edges=True)
                        else:
                            # Standard Gaussian blur
                            mask_soft = cv2.GaussianBlur(mask_float, blur_kernel, 0)
                    except Exception:
                        # Fallback to adaptive kernel if classification fails
                        mask_soft = cv2.GaussianBlur(mask_float, blur_kernel, 0)
                else:
                    # No seed point - use adaptive kernel only
                    mask_soft = cv2.GaussianBlur(mask_float, blur_kernel, 0)
            except Exception as e:
                # Fallback to legacy blur on any error
                import logging
                logging.warning(f"Adaptive blur failed, using legacy: {e}")
                mask_soft = cv2.GaussianBlur(mask_float, ColorizerConfig.BLUR_KERNEL_SIZE, 0)
        else:
            # Legacy mode: fixed blur
            mask_soft = cv2.GaussianBlur(mask_float, ColorizerConfig.BLUR_KERNEL_SIZE, 0)
            
        # HARD SAFETY RULE: Absolutely no bleeding outside original final_mask
        # Forces alpha = 0 where original mask = 0
        mask_soft = mask_soft * mask_float
        
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Prepare Target Color
        target_rgb = ColorTransferEngine.hex_to_rgb(target_color_hex)
        
        # 3. LAB Color Transfer with Caching
        img_float = image_rgb.astype(np.float32) / 255.0
        
        # Cache LAB conversion if possible
        cache_key = f"lab_conversion_{id(image_rgb)}"
        if cache_key in st.session_state:
            img_lab = st.session_state[cache_key]
        else:
            img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
            # Cache for reuse (helps with multiple layers)
            st.session_state[cache_key] = img_lab
        
        L, A, B = cv2.split(img_lab)
        
        # Target color in LAB (cached via get_target_lab)
        target_L, target_a, target_b = ColorTransferEngine.get_target_lab(target_color_hex)
        
        # Calculate mean lightness of the original wall to preserve texture
        mask_bool = mask_soft > 0.1
        if np.any(mask_bool):
            mean_L = np.mean(L[mask_bool])
        else:
            mean_L = target_L
            
        new_L = np.clip(target_L + (L - mean_L), 0, 100)
        
        # Preserve original Luminance contrast, swap A/B channels
        new_A = np.full_like(A, target_a)
        new_B = np.full_like(B, target_b)
        
        new_lab = cv2.merge([new_L, new_A, new_B])
        recolored_rgb = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)
        
        # 4. Blend based on mask
        result_float = (recolored_rgb * mask_3ch) + (img_float * (1.0 - mask_3ch))
        
        result_uint8 = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
        
        return result_uint8

    @staticmethod
    @st.cache_data
    def get_target_lab(color_hex):
        """Pre-calculate and cache the LAB channels for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 0]), float(lab[0, 0, 1]), float(lab[0, 0, 2])

    @staticmethod
    def composite_multiple_layers(image_rgb, masks_data):
        """
        ULTRA-STABLE Single-Pass Compositor with Smart Caching.
        
        Optimized to handle 'Add Layer' operations incrementally.
        only re-calculates the new layer on top of the cached previous state.
        """
        if not masks_data:
            return image_rgb.copy()

        h, w = image_rgb.shape[:2]
        
        # --- CACHING LOGIC ---
        # We need to decide: Start from scratch OR Start from cached state?
        
        # 1. Access Base LAB (Always needed for L-channel reference)
        l_cache_key = "global_base_lab"
        
        if (l_cache_key not in st.session_state or 
            st.session_state.get("lab_cache_id") != id(image_rgb) or
            st.session_state.get("lab_cache_dim") != (h, w)):
            
            img_f = image_rgb.astype(np.float32, copy=False) / 255.0
            img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
            L, A, B = cv2.split(img_lab)
            st.session_state[l_cache_key] = (L, A, B)
            st.session_state["lab_cache_id"] = id(image_rgb)
            st.session_state["lab_cache_dim"] = (h, w)
            
            # Reset composite cache if base image changed
            st.session_state["comp_cache_state"] = None
            st.session_state["comp_cache_len"] = 0
            st.session_state["comp_cache_last_id"] = None
        
        # Load Base
        base_L, base_A, base_B = st.session_state[l_cache_key]
        
        # --- PRE-COMPUTE GROUPED MEAN LIGHTNESS ---
        # To prevent 'white bleaches' when multiple masks of the same color overlap or are small,
        # we compute a single mean_L for the UNION of all masks sharing the same color.
        color_masks = {}
        for data in masks_data:
            color = data.get('color')
            mask = data.get('mask')
            if color and mask is not None:
                if sparse.issparse(mask): m = mask.toarray().astype(np.float32)
                else: m = mask.astype(np.float32)
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                if color not in color_masks:
                    color_masks[color] = np.zeros((h, w), dtype=np.float32)
                color_masks[color] = np.maximum(color_masks[color], m)
                
        current_means = {}
        for color, combined_mask in color_masks.items():
            m_bool = combined_mask > 0.1
            if np.any(m_bool):
                current_means[color] = np.mean(base_L[m_bool])
            else:
                current_means[color] = None
        
        # 2. Check for Incremental Update
        cached_state = st.session_state.get("comp_cache_state")
        cached_len = st.session_state.get("comp_cache_len", 0)
        cached_means = st.session_state.get("comp_cache_means", {})
        
        start_index = 0
        curr_A = base_A.copy()
        curr_B = base_B.copy()
        curr_L_mod = base_L.copy() 

        can_use_cache = False
        
        if cached_state is not None and len(masks_data) > cached_len:
            if cached_len > 0:
                last_cached_mask = masks_data[cached_len-1]
                if id(last_cached_mask) == st.session_state.get("comp_cache_last_id"):
                     can_use_cache = True
            else:
                can_use_cache = True
                
        # Invalidate cache if grouped means changed significantly (e.g., > 1.0)
        if can_use_cache:
            for color, mean_val in current_means.items():
                cached_mean = cached_means.get(color)
                if mean_val is not None and cached_mean is not None:
                    if abs(mean_val - cached_mean) > 1.0:
                        can_use_cache = False
                        break
                elif mean_val != cached_mean:
                    can_use_cache = False
                    break
        
        if can_use_cache:
            c_L, c_A, c_B = cached_state
            curr_L_mod = c_L.copy()
            curr_A = c_A.copy()
            curr_B = c_B.copy()
            start_index = cached_len
        else:
            pass

        # 3. Cumulative A/B Blending
        for i in range(start_index, len(masks_data)):
            data = masks_data[i]
            mask = data['mask']
            color_hex = data.get('color')
            if not color_hex: continue
            
            # ⚡ MEMORY OPTIMIZATION: Decompress if sparse
            if sparse.issparse(mask):
                mask = mask.toarray()
            
            target_L, target_a, target_b = ColorTransferEngine.get_target_lab(color_hex)
            
            # Robust preparation
            if mask.shape[:2] != (h, w):
                mask_uint8 = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_f = mask_uint8.astype(np.float32)
            else:
                mask_f = mask.astype(np.float32)

            if mask_f.max() > 1.0: mask_f /= 255.0
            
            # BLUR / REFINEMENT
            refinement = data.get('refinement', 0)
            if refinement != 0:
                k_size = abs(refinement) * 2 + 1
                refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                if refinement > 0: mask_f = cv2.dilate(mask_f, refine_kernel)
                else: mask_f = cv2.erode(mask_f, refine_kernel)

            user_soft = data.get('softness', 0)
            if user_soft > 0:
                k_size = user_soft * 4 + 1
                blur_val = (k_size, k_size)
            else:
                blur_val = ColorizerConfig.BLUR_KERNEL_SIZE
            
            if ColorizerConfig.DILATION_ITERATIONS > 0:
                kernel = np.ones(ColorizerConfig.DILATION_KERNEL_SIZE, np.uint8)
                mask_dilated = cv2.dilate(mask_f, kernel, iterations=ColorizerConfig.DILATION_ITERATIONS)
            else:
                mask_dilated = mask_f
                
            if user_soft == 0:
                # Auto-selected wall: fix the "white strip" edge halo problem.
                # Expand the mask by 1 pixel so the paint fully covers the boundary,
                # then apply a tight anti-aliasing blur to keep edges smooth but sharp.
                mask_expanded = cv2.dilate(mask_dilated, np.ones((3,3), np.uint8), iterations=1)
                mask_soft = cv2.GaussianBlur(mask_expanded, (3,3), 0)
            else:
                # Manual brush stroke: use the requested softness
                mask_soft = cv2.GaussianBlur(mask_dilated, blur_val, 0)
                
            # --- DEBUG VISUALIZATION ---
            import os
            debug_dir = r"d:\paint\debug_masks"
            if os.path.exists(debug_dir):
                cv2.imwrite(os.path.join(debug_dir, f"01_original_mask_{i}.png"), (mask_f * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(debug_dir, f"02_expanded_mask_{i}.png"), (mask_expanded * 255).astype(np.uint8) if user_soft == 0 else (mask_dilated * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(debug_dir, f"03_feathered_mask_{i}.png"), (mask_soft * 255).astype(np.uint8))
            # ---------------------------
            
            # Use the robust grouped mean_L for this color
            mean_L = current_means.get(color_hex)
            if mean_L is None:
                mean_L = target_L
                
            # Shift lightness based on target color
            adjusted_L = np.clip(target_L + (base_L - mean_L), 0, 100)
            
            # L-Channel Adjustment
            finish = data.get('finish', 'Standard')
            layer_L = adjusted_L.copy()
            if finish != 'Standard':
                if finish == 'Matte': layer_L = np.clip(adjusted_L * 0.85 + 7, 0, 100)
                elif finish == 'Gloss': layer_L = np.clip((adjusted_L - 50) * 1.35 + 50, 0, 100)
                elif finish == 'Satin': layer_L = np.clip((adjusted_L - 50) * 1.15 + 50, 0, 100)
                elif finish == 'Texture': 
                     layer_L = np.clip((adjusted_L * (target_L / 70.0)), 0, 100)
            
            curr_L_mod = (layer_L * mask_soft) + (curr_L_mod * (1.0 - mask_soft))

            curr_A = (target_a * mask_soft) + (curr_A * (1.0 - mask_soft))
            curr_B = (target_b * mask_soft) + (curr_B * (1.0 - mask_soft))
            
            # --- DEBUG VISUALIZATION ---
            if os.path.exists(debug_dir):
                temp_lab = cv2.merge([curr_L_mod, curr_A, curr_B])
                temp_rgb = cv2.cvtColor(temp_lab, cv2.COLOR_Lab2RGB)
                cv2.imwrite(os.path.join(debug_dir, f"04_composited_{i}.png"), cv2.cvtColor(np.clip(temp_rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            # ---------------------------

        # 4. Save Cache
        st.session_state["comp_cache_state"] = (curr_L_mod.copy(), curr_A.copy(), curr_B.copy())
        st.session_state["comp_cache_len"] = len(masks_data)
        st.session_state["comp_cache_means"] = current_means
        if masks_data:
            st.session_state["comp_cache_last_id"] = id(masks_data[-1])
        else:
             st.session_state["comp_cache_last_id"] = None

        # 5. Final Conversion
        final_lab = cv2.merge([curr_L_mod, curr_A, curr_B])
        final_rgb = cv2.cvtColor(final_lab, cv2.COLOR_Lab2RGB)
        
        return np.clip(final_rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_texture(image_rgb, mask, texture_rgb, opacity=0.8):
        """
        Apply a texture with blending to simulate surface material.
        """
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Create Smooth Mask
        mask_float = mask.astype(np.float32)
        mask_soft = cv2.GaussianBlur(mask_float, ColorizerConfig.BLUR_KERNEL_SIZE, 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Tile Texture to fill image
        h, w, c = image_rgb.shape
        th, tw, tc = texture_rgb.shape
        
        # Resize texture if too large to keep pattern visible
        if max(th, tw) > ColorizerConfig.MAX_TEXTURE_SIZE:
            scale = ColorizerConfig.MAX_TEXTURE_SIZE / max(th, tw)
            texture_rgb = cv2.resize(texture_rgb, (0, 0), fx=scale, fy=scale)
            th, tw, tc = texture_rgb.shape
            
        tiled_texture = np.zeros_like(image_rgb)
        
        for i in range(0, h, th):
            for j in range(0, w, tw):
                # Calculate available space
                curr_h = min(th, h - i)
                curr_w = min(tw, w - j)
                tiled_texture[i:i+curr_h, j:j+curr_w] = texture_rgb[:curr_h, :curr_w]
                
        # 3. Blend Texture (Multiply/Overlay approach)
        # Simple Approach: Multiply original L with Texture
        
        img_float = image_rgb.astype(np.float32) / 255.0
        tex_float = tiled_texture.astype(np.float32) / 255.0
        
        # Luminosity preservation:
        # Result = Texture * Original_Luminance
        # This makes the texture look shadowed by the room's lighting.
        
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(img_lab)
        
        # Blend: use texture color but keep original lightness structure
        # Optionally mix texture's own lightness with original lightness
        
        # Simplified: Alpha Blend texture over image, but modulated by mask
        # To make it look "on the wall", we ideally want:
        # Out = Texture * (Original_Gray) * 2.0 (Overlay-ish)
        
        gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        gray_3ch = np.stack([gray] * 3, axis=-1)
        
        # Hard Light / Multiply simulation
        blended = tex_float * gray_3ch * ColorizerConfig.TEXTURE_BRIGHTNESS_BOOST # Boost brightness slightly
        
        blended = np.clip(blended, 0, 1.0)
        
        # 4. Composite
        # Result = (Blended * Mask * Opacity) + (Original * (1 - Mask*Opacity))
        
        final_mask = mask_3ch * opacity
        output = (blended * final_mask) + (img_float * (1.0 - final_mask))
        
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)
