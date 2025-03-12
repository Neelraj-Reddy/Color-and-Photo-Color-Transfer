# Paper Link : https://users.cs.northwestern.edu/~bgooch/PDFs/ColorTransfer.pdf


This paper describes the **statistical color transfer** method used to adjust the colors of one image (synthetic/source) to match the look and feel of another image (target). Let’s break it down step by step.

---

## **1. What Is Happening Here?**
The idea is to transfer **color statistics** from a target image to a source image. The process follows these steps:

1. **Convert both images to lαβ space.**  
   - The lαβ space is chosen because its color channels are **decorrelated**, making it easier to manipulate them independently.
  
2. **Compute the mean and standard deviation** for each color channel (**l, α, β**) in both the source and target images.

3. **Normalize the source image** by subtracting its mean and scaling by its standard deviation:
   \[
   l' = \frac{l - \mu_{l,s}}{\sigma_{l,s}}
   \]
   \[
   α' = \frac{α - \mu_{α,s}}{\sigma_{α,s}}
   \]
   \[
   β' = \frac{β - \mu_{β,s}}{\sigma_{β,s}}
   \]
   This ensures that the transformed source image has **zero mean and unit variance** in lαβ space.

4. **Match the target image’s color distribution** by applying the mean and standard deviation of the target:
   \[
   l^* = l' \cdot \sigma_{l,t} + \mu_{l,t}
   \]
   \[
   α^* = α' \cdot \sigma_{α,t} + \mu_{α,t}
   \]
   \[
   β^* = β' \cdot \sigma_{β,t} + \mu_{β,t}
   \]
   This scales the normalized source image to have the same **mean and standard deviation** as the target image.

5. **Convert the modified lαβ values back to RGB.**  
   - This involves going from **lαβ → LMS → XYZ → RGB** using the inverse transformations.

---

## **2. Why Does This Work?**
- The **mean** shift aligns the **overall color tone** of the source image to the target image.
- Scaling by **standard deviation** adjusts the **color contrast** to match that of the target.
- Since **lαβ space** is decorrelated, these operations affect each color channel **independently**, making the transformation more natural.

---

## **3. Handling Multiple Color Clusters**
The paper also notes that simple mean-variance matching might fail if:
- The source and target images have **different dominant colors** (e.g., source has a lot of grass, target has more sky).
- The method assumes **global** color statistics, but colors may vary across different regions.

### **Solution: Cluster-Based Color Matching**
- Instead of treating the whole image as one unit, the image can be split into **clusters** (e.g., sky, grass).
- Compute **mean and standard deviation** for each cluster separately.
- Assign each pixel in the source image to the **nearest cluster** in the target image using **normalized distances**.
- Blend color-mapped pixels using weights **inversely proportional to distance** from cluster centers.

This **localized approach** improves quality when the images contain **different regions with different color distributions**.

---

## **4. Extending Beyond Mean and Standard Deviation**
- Higher-order statistics like **skew (asymmetry of color distribution)** and **kurtosis (how heavy the color tails are)** could be used.
- Adjusting skew/kurtosis might further refine color transfers, making the result **more visually accurate**.

---

## **5. Summary**
- **Color transfer is done by shifting and scaling statistics (mean & standard deviation) in lαβ space.**
- **Handling multiple clusters** improves performance when source and target images have different dominant colors.
- **Higher-order statistics (skew, kurtosis)** could further enhance the transformation.

