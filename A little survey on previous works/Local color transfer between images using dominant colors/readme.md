# Local Color Transfer Using Dominant Colors

## **Overview**
This method transfers color styles from a target image to a source image by segmenting images into dominant color regions and performing localized color transfer. It consists of three major steps:

1. **Soft Segmentation** - Extract dominant colors and segment the image.
2. **Region Matching** - Match corresponding regions in the source and target.
3. **Local Color Transfer** - Apply modified Reinhard’s color transfer method.

---

## **1. Soft Segmentation**
To transfer complex color styles, the source and target images are first segmented based on their dominant colors.

### **1.1 Estimating Dominant Colors**
- The **HSV color space** is used since the hue component efficiently defines color.
- A **grid-based approach** is used instead of Mean-Shift to find high-density regions.
- **Outlier removal** is done using the **3-sigma rule**:

$$
\tau = \mu - 3\sigma
$$

where:
- \( \mu \) = mean number of pixels in each grid,
- \( \sigma \) = standard deviation.

### **1.2 Cost-Volume Segmentation**
- The image is converted to **CIELab space** for better perceptual accuracy.
- Cost-volume filtering is used for segmentation by measuring perceptual color difference **ΔE94**:

$$
\Delta E_{94} = \sqrt{ \left(\frac{\Delta L}{K_L}\right)^2 + \left(\frac{\Delta AB}{1 + K_1 AB1}\right)^2 + \left(\frac{\Delta H}{1 + K_2 AB1}\right)^2 }
$$

where:
- \( \Delta L = L_1 - L_2 \) (lightness difference),
- \( \Delta AB = AB_1 - AB_2 \) (chroma difference),
- \( \Delta H \) = hue difference,
- Constants: \( K_L = 2 \), \( K_1 = 0.048 \), \( K_2 = 0.014 \).

### **1.3 Guided Feathering**
- To avoid hard segment boundaries, a **Gaussian filter** is applied for smooth blending.

---

## **2. Region Matching**
After segmentation, each region in the source image is matched to a corresponding region in the target.

### **2.1 Feature Extraction**
Each region is characterized using:
1. **Visual Saliency (S)**  
   - Saliency is computed using spectral residual analysis:

   $$
   S = || I_{\mu} - I_{\omega hc}(x,y) ||
   $$

   where \( I_{\mu} \) is the mean CIELab value and \( I_{\omega hc} \) is a Gaussian-blurred version of the image.

2. **Luminance (L)**  
   - Mean luminance of the region.

3. **Pixel Ratio (R)**  
   - Ratio of the region’s size to the total image size.

### **2.2 Matching Function**
Regions are matched based on weighted distances between features:

$$
f(i) = \arg \min_j \sqrt{ w_S (S_i - S_j)^2 + w_L (L_i - L_j)^2 + w_R (R_i - R_j)^2 }
$$

where \( w_S, w_L, w_R \) are adjustable weight factors.

---

## **3. Local Color Transfer**
Color transfer is performed in **CIELab space** using a modified version of Reinhard’s method.

### **3.1 Reinhard’s Color Transfer**
For each matched region, color is transferred using:

$$
a' = \frac{\sigma_j}{\sigma_i} (a_i - \mu_i) + \mu_j
$$

$$
b' = \frac{\sigma_j}{\sigma_i} (b_i - \mu_i) + \mu_j
$$

where:
- \( \mu \) = mean of the chromatic channels (A, B),
- \( \sigma \) = standard deviation of chromatic channels.

### **3.2 Local Gamma Correction for Luminance**
Since luminance differences can cause unnatural results, a **local gamma correction** is applied:

$$
L' = L^{\gamma_i}
$$

where the gamma correction value \( \gamma_i \) is:

$$
\gamma_i = |\beta_i + \alpha (\mu_{Ls} - \mu_{Lt})|
$$

with:

$$
\beta_i = 1 + 2 (\mu_{li} - \mu_{lj})
$$

$$
\alpha_i = \exp \left( \frac{|L_s - L_t|}{\mu_{li} / \mu_{Ls} - \mu_{lj} / \mu_{Lt}} \right)
$$

This ensures a natural balance of luminance.

---

## **Final Pipeline**
1. **Convert images to HSV → Extract dominant colors using grid-based clustering.**
2. **Convert to CIELab → Compute cost-volume segmentation.**
3. **Compute saliency, luminance, and pixel ratio for each region.**
4. **Match regions based on weighted feature distances.**
5. **Apply Reinhard’s color transfer in CIELab space per region.**
6. **Use local gamma correction for luminance balance.**
7. **Convert back to RGB and output the final result.**

---

## **Advantages of the Proposed Method**
✔ **Handles complex color styles better than global color transfer methods.**  
✔ **Soft segmentation ensures smoother region blending.**  
✔ **Saliency-based region matching improves natural appearance.**  
✔ **Luminance balance prevents over-saturation and unnatural lighting.**  

---

## **References**
1. Yoo et al., "Local Color Transfer between Images Using Dominant Colors," *Journal of Electronic Imaging*, 2013.  
2. Reinhard et al., "Color Transfer between Images," *IEEE Transactions on Visualization and Computer Graphics*, 2001.
