import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#############################################
# Processing Functions (Example: LAB-based PCA Transfer)
#############################################

def sqrtm(matrix, method="svd"):
    # Simple square-root using eigen decomposition.
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def compute_mean_and_cov(image):
    reshaped = image.reshape(-1, 3).astype(np.float32)
    mean = np.mean(reshaped, axis=0)
    cov = np.cov(reshaped, rowvar=False)
    return mean, cov

def pca_transfer_lab(target, reference):
    """
    Applies PCA-based transfer on the chrominance (A and B) channels in LAB.
    Luminance (L) is preserved from the target.
    """
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_t = target_lab[:, :, 0]
    AB_t = target_lab[:, :, 1:3]
    AB_r = ref_lab[:, :, 1:3]
    AB_t /= 255.0
    AB_r /= 255.0
    AB_t_flat = AB_t.reshape(-1, 2)
    AB_r_flat = AB_r.reshape(-1, 2)
    mu_t = np.mean(AB_t_flat, axis=0)
    mu_r = np.mean(AB_r_flat, axis=0)
    cov_t = np.cov(AB_t_flat, rowvar=False)
    cov_r = np.cov(AB_r_flat, rowvar=False)
    sqrt_cov_t = sqrtm(cov_t, method="eigen")
    sqrt_cov_r = sqrtm(cov_r, method="eigen")
    transform = sqrt_cov_r @ np.linalg.inv(sqrt_cov_t)
    AB_transformed = ((AB_t_flat - mu_t) @ transform.T) + mu_r
    AB_transformed = np.clip(AB_transformed, 0, 1).reshape(AB_t.shape)
    AB_transformed = (AB_transformed * 255).astype(np.uint8)
    L_t = np.clip(L_t, 0, 255).astype(np.uint8)
    lab_transferred = cv2.merge((L_t, AB_transformed[:, :, 0], AB_transformed[:, :, 1]))
    return cv2.cvtColor(lab_transferred, cv2.COLOR_LAB2BGR)

def process_sequential(target, ref_list):
    """
    Sequentially applies the color transfer.
    target: initial target image (BGR).
    ref_list: list of reference images (BGR) in the desired order.
    Returns a list of tuples: (label, image) with intermediate outputs.
    """
    results = [("Target", target.copy())]
    current = target.copy()
    for i, ref in enumerate(ref_list, start=1):
        if ref.shape != current.shape:
            ref = cv2.resize(ref, (current.shape[1], current.shape[0]))
        current = pca_transfer_lab(current, ref)
        results.append((f"Output {i}", current.copy()))
    return results


#############################################
# Utility / Processing Functions
#############################################

def cv2_to_tk(cv_img, maxsize=(500,500)):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail(maxsize)
    return ImageTk.PhotoImage(pil_img)

def create_polygon_mask(points, img_shape, disp_dims):
    """
    Converts polygon points from canvas coordinates to image coordinates.
    disp_dims: (display_width, display_height)
    img_shape: (img_height, img_width)
    """
    disp_w, disp_h = disp_dims
    img_h, img_w = img_shape
    scale_x = img_w / disp_w
    scale_y = img_h / disp_h
    scaled_points = [(int(x * scale_x), int(y * scale_y)) for (x, y) in points]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)

def get_top_colors(image, k=10):
    """
    Uses KMeans to extract top k colors from the image.
    Returns a list of colors as lists [B, G, R]. If k-means fails to find enough clusters,
    returns the unique colors found.
    """
    Z = image.reshape((-1, 3)).astype(np.float32)
    try:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(Z)
        centers = kmeans.cluster_centers_.astype(np.uint8).tolist()
        return centers
    except Exception as e:
        unique_colors = np.unique(Z, axis=0)
        return unique_colors.tolist()

def create_feathered_alpha(mask, feather=15):
    """
    Computes a smooth alpha mask using a distance transform.
    """
    mask_uint = mask.astype(np.uint8)
    inv_mask = cv2.bitwise_not(mask_uint)
    dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
    if np.any(mask_uint == 255):
        max_val = np.max(dist[mask_uint == 255])
    else:
        max_val = 1
    alpha = 1 - (dist / (max_val + 1e-5))
    alpha = cv2.GaussianBlur(alpha, (feather, feather), 0)
    return np.clip(alpha, 0, 1)

def apply_tint_feathering(img_bgr, mask, tint_color, blend=0.6, feather=15):
    """
    Blends tint_color into img_bgr over the region defined by mask,
    using a feathered alpha for smooth transitions.
    """
    alpha = create_feathered_alpha(mask, feather=feather)
    tint_img = np.full_like(img_bgr, tint_color, dtype=np.float32)
    img_float = img_bgr.astype(np.float32)
    blended = (1 - blend * alpha[..., None]) * img_float + (blend * alpha[..., None]) * tint_img
    return np.clip(blended, 0, 255).astype(np.uint8)

def compute_mean_color(img_bgr):
    """
    Computes the mean [B, G, R] of the image.
    """
    return np.mean(img_bgr.reshape(-1,3), axis=0).tolist()

def apply_local_mapping(target_img, target_mask, mapping, threshold=40):
    """
    For each pixel in target_img within target_mask, if its color is close to one of the keys in mapping,
    then blend that pixel toward the mapped color.
    'mapping' is a dict with keys and values as tuples: {target_color: ref_color}.
    """
    out = target_img.copy().astype(np.float32)
    indices = np.where(target_mask)
    for y, x in zip(*indices):
        pixel = target_img[y, x].tolist()
        for t_color, r_color in mapping.items():
            d = np.linalg.norm(np.array(pixel, dtype=np.float32) - np.array(t_color, dtype=np.float32))
            if d < threshold:
                w = 1 - (d / threshold)
                new_val = (1 - w) * np.array(pixel) + w * np.array(r_color)
                out[y, x] = new_val
                break
    return np.clip(out, 0, 255).astype(np.uint8)

#############################################
# Interactive UI Application
#############################################

class InteractiveLocalMappingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interactive Local Color Mapping")
        self.geometry("1400x900")
        
        # Images (BGR)
        self.target_img = None
        self.ref_img = None
        self.processed_img = None  # after global transfer
        
        # Polygons (canvas coordinates)
        self.target_poly_points = []
        self.ref_poly_points = []
        self.target_poly_id = None
        self.ref_poly_id = None
        
        # Displayed dimensions (for scaling)
        self.target_disp_dims = None
        self.ref_disp_dims = None
        
        # Local masks
        self.target_mask = None
        self.ref_mask = None
        
        # Dominant colors extracted from local regions.
        self.target_local_colors = []
        self.ref_local_colors = []
        
        # Mapping dictionary: keys and values as tuples: (target_color) -> (ref_color)
        self.mapping = {}
        
        # Temporary storage for mapping: when selecting, store target color first.
        self.temp_target_color = None
        
        self.create_widgets()
    
    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(padx=10, pady=5, fill=tk.X)
        
        tk.Button(top_frame, text="Load Target Image", command=self.load_target).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load Reference Image", command=self.load_reference).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Apply Global Transfer", command=self.apply_global_transfer).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Finish Target Polygon", command=self.finish_target_polygon).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Finish Reference Polygon", command=self.finish_ref_polygon).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Undo Target Point", command=self.undo_target_point).pack(side=tk.LEFT, padx=5)  # Added button
        tk.Button(top_frame, text="Undo Reference Point", command=self.undo_ref_point).pack(side=tk.LEFT, padx=5)  # Added button
        tk.Button(top_frame, text="Extract Local Colors", command=self.extract_local_colors).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Apply Local Mapping", command=self.apply_local_mapping).pack(side=tk.LEFT, padx=5)
        
        main_frame = tk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Left: Target canvas for polygon selection.
        target_frame = tk.Frame(main_frame)
        target_frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(target_frame, text="Target Image").pack()
        self.target_canvas = tk.Canvas(target_frame, bg="gray", width=600, height=600)
        self.target_canvas.pack()
        self.target_canvas.bind("<Button-1>", self.on_target_click)
        
        # Right: Reference canvas for polygon selection.
        ref_frame = tk.Frame(main_frame)
        ref_frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(ref_frame, text="Reference Image").pack()
        self.ref_canvas = tk.Canvas(ref_frame, bg="gray", width=600, height=600)
        self.ref_canvas.pack()
        self.ref_canvas.bind("<Button-1>", self.on_ref_click)
        
        # Bottom: Local palettes and mapping display.
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(padx=10, pady=5, fill=tk.X)
        
        tk.Label(bottom_frame, text="Target Local Colors:").grid(row=0, column=0, sticky="w")
        self.target_palette_frame = tk.Frame(bottom_frame)
        self.target_palette_frame.grid(row=1, column=0, padx=5)
        
        tk.Label(bottom_frame, text="Reference Local Colors:").grid(row=0, column=1, sticky="w")
        self.ref_palette_frame = tk.Frame(bottom_frame)
        self.ref_palette_frame.grid(row=1, column=1, padx=5)
        
        tk.Label(bottom_frame, text="Mappings (click target then reference):").grid(row=0, column=2, sticky="w")
        self.mapping_frame = tk.Frame(bottom_frame)
        self.mapping_frame.grid(row=1, column=2, padx=5)
        
        self.status_label = tk.Label(self, text="Load images to begin...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
    
    def load_target(self):
        path = filedialog.askopenfilename(title="Select Target Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Failed to load target image.")
                return
            self.target_img = img
            self.processed_img = img.copy()
            self.display_on_canvas(self.target_canvas, self.processed_img)
            self.target_disp_dims = (self.target_canvas.winfo_width(), self.target_canvas.winfo_height())
            self.target_poly_points = []
            if self.target_poly_id:
                self.target_canvas.delete(self.target_poly_id)
                self.target_poly_id = None
            self.status_label.config(text="Target image loaded. Now load reference image.")
    
    def load_reference(self):
        path = filedialog.askopenfilename(title="Select Reference Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Failed to load reference image.")
                return
            self.ref_img = img
            self.display_on_canvas(self.ref_canvas, self.ref_img)
            self.ref_disp_dims = (self.ref_canvas.winfo_width(), self.ref_canvas.winfo_height())
            self.status_label.config(text="Reference image loaded. Click 'Apply Global Transfer' when ready.")
    
    def apply_global_transfer(self):
        if self.target_img is None or self.ref_img is None:
            messagebox.showwarning("Missing Images", "Please load both target and reference images.")
            return
        self.processed_img = pca_transfer_lab(self.target_img, self.ref_img)
        self.display_on_canvas(self.target_canvas, self.processed_img)
        self.status_label.config(text="Global transfer applied. Now draw polygons on both images.")
    
    def on_target_click(self, event):
        self.target_poly_points.append((event.x, event.y))
        r = 3
        self.target_canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red")
        if self.target_poly_points:
            if self.target_poly_id:
                self.target_canvas.delete(self.target_poly_id)
            self.target_poly_id = self.target_canvas.create_polygon(self.target_poly_points, outline="red", fill="", width=2)
            self.status_label.config(text=f"Target polygon: {self.target_poly_points}")
    
    def on_ref_click(self, event):
        self.ref_poly_points.append((event.x, event.y))
        r = 3
        self.ref_canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="blue")
        if self.ref_poly_points:
            if self.ref_poly_id:
                self.ref_canvas.delete(self.ref_poly_id)
            self.ref_poly_id = self.ref_canvas.create_polygon(self.ref_poly_points, outline="blue", fill="", width=2)
            self.status_label.config(text=f"Reference polygon: {self.ref_poly_points}")
    
    def finish_target_polygon(self):
        if len(self.target_poly_points) < 3:
            messagebox.showwarning("Insufficient Points", "Please select at least 3 points on the target image.")
            return
        self.status_label.config(text="Target polygon finished.")
    
    def finish_ref_polygon(self):
        if len(self.ref_poly_points) < 3:
            messagebox.showwarning("Insufficient Points", "Please select at least 3 points on the reference image.")
            return
        self.status_label.config(text="Reference polygon finished.")
    
    def undo_target_point(self):
        """
        Undo the last point added to the target polygon.
        """
        if self.target_poly_points:
            self.target_poly_points.pop()
            if self.target_poly_id:
                self.target_canvas.delete(self.target_poly_id)
            if self.target_poly_points:
                self.target_poly_id = self.target_canvas.create_polygon(self.target_poly_points, outline="red", fill="", width=2)
            self.status_label.config(text=f"Target polygon updated: {self.target_poly_points}")
        else:
            self.status_label.config(text="No points to undo for target polygon.")

    def undo_ref_point(self):
        """
        Undo the last point added to the reference polygon.
        """
        if self.ref_poly_points:
            self.ref_poly_points.pop()
            if self.ref_poly_id:
                self.ref_canvas.delete(self.ref_poly_id)
            if self.ref_poly_points:
                self.ref_poly_id = self.ref_canvas.create_polygon(self.ref_poly_points, outline="blue", fill="", width=2)
            self.status_label.config(text=f"Reference polygon updated: {self.ref_poly_points}")
        else:
            self.status_label.config(text="No points to undo for reference polygon.")
    
    def extract_local_colors(self):
        if len(self.target_poly_points) < 3 or len(self.ref_poly_points) < 3:
            messagebox.showwarning("Missing Polygon", "Please finish drawing polygons on both images.")
            return
        self.target_mask = create_polygon_mask(self.target_poly_points, self.processed_img.shape[:2], self.target_disp_dims)
        self.ref_mask = create_polygon_mask(self.ref_poly_points, self.ref_img.shape[:2], self.ref_disp_dims)
        self.target_local_colors = extract_dominant_colors(self.processed_img, self.target_mask, n_colors=4)
        self.ref_local_colors = extract_dominant_colors(self.ref_img, self.ref_mask, n_colors=4)
        self.display_local_palettes()
        self.mapping = {}
        self.status_label.config(text="Local colors extracted. Click a target swatch then a reference swatch to map them.")
    
    def display_local_palettes(self):
        for widget in self.target_palette_frame.winfo_children():
            widget.destroy()
        for color in self.target_local_colors:
            hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            lbl = tk.Label(self.target_palette_frame, bg=hex_color, width=4, height=2, relief=tk.RAISED, bd=2)
            lbl.pack(side=tk.LEFT, padx=5)
            lbl.bind("<Button-1>", lambda e, col=color: self.on_target_color_select(col))
        
        for widget in self.ref_palette_frame.winfo_children():
            widget.destroy()
        for color in self.ref_local_colors:
            hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            lbl = tk.Label(self.ref_palette_frame, bg=hex_color, width=4, height=2, relief=tk.RAISED, bd=2)
            lbl.pack(side=tk.LEFT, padx=5)
            lbl.bind("<Button-1>", lambda e, col=color: self.on_ref_color_select(col))
        
        for widget in self.mapping_frame.winfo_children():
            widget.destroy()
    
    def on_target_color_select(self, color):
        # Convert list to tuple for hashing.
        self.temp_target_color = tuple(color)
        self.status_label.config(text=f"Selected target color: {self.temp_target_color}. Now select corresponding reference color.")
    
    def on_ref_color_select(self, color):
        if not hasattr(self, 'temp_target_color') or self.temp_target_color is None:
            self.status_label.config(text="Select a target color first.")
            return
        self.mapping[self.temp_target_color] = tuple(color)
        # Display mapping.
        for widget in self.mapping_frame.winfo_children():
            widget.destroy()
        mapping_text = f"Mapping: {self.temp_target_color} -> {tuple(color)}"
        tk.Label(self.mapping_frame, text=mapping_text).pack()
        self.status_label.config(text=f"Mapping added: {self.temp_target_color} -> {tuple(color)}.")
        self.temp_target_color = None
    
    def apply_local_mapping(self):
        if not self.mapping:
            messagebox.showwarning("No Mapping", "Please map at least one color pair.")
            return
        if self.target_mask is None:
            messagebox.showerror("Error", "Local target mask not available.")
            return
        new_img = apply_local_mapping(self.processed_img, self.target_mask, self.mapping, threshold=40)
        self.processed_img = new_img
        self.display_on_canvas(self.target_canvas, self.processed_img)
        self.status_label.config(text="Local mapping applied.")
    
    def display_on_canvas(self, canvas, cv_img):
        photo = cv2_to_tk(cv_img, maxsize=(canvas.winfo_width(), canvas.winfo_height()))
        canvas.photo = photo
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2, image=photo)
    
#############################################
# Additional Functions
#############################################

def extract_dominant_colors(img, mask, n_colors=8):
    pixels = img[mask].reshape(-1, 3)
    if pixels.shape[0] < n_colors:
        return []
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8).tolist()
    return colors

#############################################
# Main
#############################################

if __name__ == '__main__':
    def get_top_colors(image, k=10):
        Z = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(Z)
        centers = kmeans.cluster_centers_.astype(np.uint8).tolist()
        return centers
    
    app = InteractiveLocalMappingApp()
    app.mainloop()


