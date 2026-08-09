Act as a Senior Python Developer and AI Forensics Specialist. Create a comprehensive, production-ready Jupyter Notebook (.ipynb) that performs multi-layer metadata inspection on a single target image to check for AI generation signatures and digital provenance.

The notebook should be organized into clear Markdown sections and executable code cells covering the following pipeline:

### Notebook Requirements & Architecture

1. Setup & Environment Dependencies:
   - Install and import necessary packages (`pillow`, `c2pa-python`, `piexif`, `matplotlib`, `pytz`).
   - Define a single global variable `TARGET_IMAGE_PATH = "sample.jpg"` at the top so it can easily be changed.

2. Section 1: C2PA & Content Credentials Verification
   - Use the `c2pa-python` SDK (`c2pa.Reader`) to extract and parse JUMBF manifest stores.
   - Inspect for digital source types (e.g., `c2pa.ai_generative_training`, `trainedAlgorithmicMedia`), claim signatures, and tool attribution (OpenAI, Google, Adobe, etc.).
   - Handle exceptions gracefully if no C2PA manifest is embedded.

3. Section 2: PNG Native Text Chunk Analysis (tEXt, zTXt, iTXt)
   - Read image metadata using PIL/Pillow (`img.info`).
   - Scan for embedded generation parameters, prompts, CFG scale, seed numbers, and sampler settings common in open-source AI generators (Automatic1111, ComfyUI, Midjourney, InvokeAI).

4. Section 3: Standard EXIF & TIFF Tag Extraction
   - Extract standard EXIF metadata using `PIL.Image.ExifTags`.
   - Parse key fields: `Software`, `Make`, `Model`, `UserComment`, `ImageDescription`, and `Artist`.
   - Match extracted values against a regex list of known AI generator keywords (e.g., "DALL-E", "Midjourney", "Stable Diffusion", "Firefly", "Flux").

5. Section 4: Raw XMP Packet & Header Parsing
   - Read the binary header bytes of the file.
   - Search for XMP namespace packets (`xmlns:xmp`, `xmlns:crs`, `stEvt:softwareAgent`).
   - Extract string matches referencing generative model signatures.

6. Section 5: Consolidated Diagnostic Report & Visual Summary
   - Display the original image side-by-side with a clean, formatted Markdown diagnostic report.
   - Generate a JSON summary object containing:
     * `c2pa_detected`: (bool)
     * `png_chunks_detected`: (bool)
     * `exif_ai_tags_detected`: (bool)
     * `xmp_ai_tags_detected`: (bool)
     * `verdict`: ("CONFIRMED AI METADATA", "SUSPICIOUS METADATA", or "NO AI METADATA FOUND")
     * `detected_tool_name`: (str or None)

Ensure every code block includes try-except blocks so that if an image lacks a specific metadata layer (e.g., a JPEG without PNG chunks), the notebook continues executing cleanly without crashing.

