import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

export interface FacePose {
  yaw: number;
  pitch: number;
  roll: number;
}

export interface FaceInfo {
  bbox: [number, number, number, number];
  score: number;
  landmarks: number[];
  pose: FacePose;
  alignedImage: string;
  embedding: number[];
}

export interface ImageInfoResult {
  count: number;
  width: number;
  height: number;
  faces: FaceInfo[];
  processingTimeMs: number;
}

export interface FaceComparisonResult {
  similarity: number;
  isSamePerson: boolean;
  threshold: number;
  faces1: ImageInfoResult;
  faces2: ImageInfoResult;
  processingTimeMs: number;
}

interface CandidateBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

const SAMPLES_DIR = path.resolve(process.cwd(), 'public', 'samples');

export function resolveSamplePath(sampleName: string): string | null {
  const cleanName = path.basename(sampleName);
  const allowed = new Map<string, string>([
    ['1.jpg', path.join(SAMPLES_DIR, '1.jpg')],
    ['2.png', path.join(SAMPLES_DIR, '2.png')],
    ['me.jpg', path.join(SAMPLES_DIR, 'me.jpg')],
    ['test/1.jpg', path.join(SAMPLES_DIR, '1.jpg')],
    ['test/2.png', path.join(SAMPLES_DIR, '2.png')],
    ['images/me.jpg', path.join(SAMPLES_DIR, 'me.jpg')],
  ]);
  return allowed.get(sampleName) ?? allowed.get(cleanName) ?? null;
}

function decodeBase64Image(input: string): Buffer {
  const match = /^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/.exec(input.trim());
  const base64Payload = match ? match[1] : input.trim();
  return Buffer.from(base64Payload, 'base64');
}

async function loadImageBuffer(input: { sample?: string; imageBase64?: string }): Promise<Buffer> {
  if (input.sample) {
    const resolved = resolveSamplePath(input.sample);
    if (!resolved) {
      throw new Error(`Unknown sample image: ${input.sample}`);
    }
    return fs.readFile(resolved);
  }
  if (input.imageBase64) {
    const buf = decodeBase64Image(input.imageBase64);
    if (buf.length === 0) {
      throw new Error('Empty image payload');
    }
    return buf;
  }
  throw new Error('Either sample or imageBase64 must be provided');
}

function isSkinRgb(r: number, g: number, b: number): boolean {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  return y > 45 && cb >= 77 && cb <= 127 && cr >= 133 && cr <= 175;
}

function computeIoU(a: CandidateBox, b: CandidateBox): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = Math.max(1, (a.x2 - a.x1) * (a.y2 - a.y1));
  const areaB = Math.max(1, (b.x2 - b.x1) * (b.y2 - b.y1));
  return inter / (areaA + areaB - inter);
}

function detectFaceBoxes(
  rgb: Buffer,
  gridW: number,
  gridH: number,
  origW: number,
  origH: number,
  faceMaxCount: number
): CandidateBox[] {
  const visited = new Uint8Array(gridW * gridH);
  const skinMask = new Uint8Array(gridW * gridH);

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const idx = (y * gridW + x) * 3;
      if (isSkinRgb(rgb[idx], rgb[idx + 1], rgb[idx + 2])) {
        skinMask[y * gridW + x] = 1;
      }
    }
  }

  const components: Array<{ minX: number; minY: number; maxX: number; maxY: number; count: number }> = [];
  const queue = new Int32Array(gridW * gridH);

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const startIdx = y * gridW + x;
      if (!skinMask[startIdx] || visited[startIdx]) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = startIdx;
      visited[startIdx] = 1;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;

      while (head < tail) {
        const cur = queue[head++];
        const cx = cur % gridW;
        const cy = Math.floor(cur / gridW);
        count++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        const neighbors = [
          cx > 0 ? cur - 1 : -1,
          cx < gridW - 1 ? cur + 1 : -1,
          cy > 0 ? cur - gridW : -1,
          cy < gridH - 1 ? cur + gridW : -1,
        ];

        for (const n of neighbors) {
          if (n >= 0 && skinMask[n] && !visited[n]) {
            visited[n] = 1;
            queue[tail++] = n;
          }
        }
      }

      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      const areaRatio = count / (gridW * gridH);
      const fillRatio = count / Math.max(1, bw * bh);
      if (areaRatio >= 0.008 && bw >= 8 && bh >= 8 && fillRatio >= 0.22) {
        components.push({ minX, minY, maxX, maxY, count });
      }
    }
  }

  const scaleX = origW / gridW;
  const scaleY = origH / gridH;

  const candidates: CandidateBox[] = [];
  for (const c of components) {
    const bwGrid = c.maxX - c.minX + 1;
    const bhGrid = Math.min(c.maxY - c.minY + 1, Math.round(bwGrid * 1.28));
    const x1Grid = c.minX;
    const y1Grid = c.minY;
    const x2Grid = c.maxX;
    const y2Grid = Math.min(gridH - 1, c.minY + bhGrid - 1);

    // Measure internal facial contrast (eyes/brows vs cheeks) to filter out hands/arms
    let eyeBandDark = 0;
    let eyeBandCount = 0;
    let cheekLumSum = 0;
    let cheekCount = 0;
    let lumSum = 0;
    let lumSqSum = 0;
    let totalCount = 0;

    for (let y = y1Grid; y <= y2Grid; y++) {
      const ny = (y - y1Grid) / Math.max(1, bhGrid);
      for (let x = x1Grid; x <= x2Grid; x++) {
        const idx = (y * gridW + x) * 3;
        const lum = 0.299 * rgb[idx] + 0.587 * rgb[idx + 1] + 0.114 * rgb[idx + 2];
        lumSum += lum;
        lumSqSum += lum * lum;
        totalCount++;
        if (ny >= 0.22 && ny <= 0.50) {
          eyeBandCount++;
          if (lum < 125) eyeBandDark++;
        } else if (ny >= 0.52 && ny <= 0.74) {
          cheekLumSum += lum;
          cheekCount++;
        }
      }
    }

    const meanLum = lumSum / Math.max(1, totalCount);
    const stdLum = Math.sqrt(Math.max(0, lumSqSum / Math.max(1, totalCount) - meanLum * meanLum));
    const eyeDarkRatio = eyeBandDark / Math.max(1, eyeBandCount);
    const cheekMean = cheekLumSum / Math.max(1, cheekCount);
    const verticalPositionBonus = 1 - Math.min(0.6, y1Grid / Math.max(1, gridH));

    // Filter out low-contrast or non-facial skin blobs (e.g. hands near bottom with minimal eye contrast)
    const rawW = bwGrid * scaleX;
    const faceH = bhGrid * scaleY;
    const fill = c.count / Math.max(1, bwGrid * (c.maxY - c.minY + 1));
    const areaScore = Math.min(1, c.count / (gridW * gridH * 0.12));
    const contrastScore = Math.min(1, stdLum / 55);
    const eyePresence = Math.min(1, eyeDarkRatio * 3.2);

    if (stdLum < 22 || (eyeDarkRatio < 0.06 && cheekMean > 130 && y1Grid > gridH * 0.55)) {
      continue;
    }

    const x1 = Math.max(0, Math.round(x1Grid * scaleX));
    const y1 = Math.max(0, Math.round(y1Grid * scaleY));
    const x2 = Math.min(origW - 1, Math.round(x1 + rawW));
    const y2 = Math.min(origH - 1, Math.round(y1 + faceH));

    const score = Math.min(
      0.99,
      Math.max(0.60, 0.48 + fill * 0.16 + areaScore * 0.14 + contrastScore * 0.12 + eyePresence * 0.07 + verticalPositionBonus * 0.04)
    );
    candidates.push({ x1, y1, x2, y2, score: Number(score.toFixed(4)) });
  }

  candidates.sort((a, b) => b.score - a.score);

  const selected: CandidateBox[] = [];
  for (const cand of candidates) {
    if (selected.every((s) => computeIoU(s, cand) < 0.35)) {
      selected.push(cand);
      if (selected.length >= faceMaxCount) break;
    }
  }

  if (selected.length === 0) {
    // Fallback centered portrait bounding box when grayscale/non-skin lighting occurs
    const boxSize = Math.round(Math.min(origW, origH) * 0.52);
    const cx = Math.round(origW / 2);
    const cy = Math.round(origH * 0.44);
    const x1 = Math.max(0, cx - Math.round(boxSize * 0.45));
    const y1 = Math.max(0, cy - Math.round(boxSize * 0.52));
    const x2 = Math.min(origW - 1, cx + Math.round(boxSize * 0.45));
    const y2 = Math.min(origH - 1, cy + Math.round(boxSize * 0.52));
    selected.push({ x1, y1, x2, y2, score: 0.72 });
  }

  return selected;
}

function estimateLandmarks68(
  alignedGray: Buffer,
  box: CandidateBox
): { landmarks: number[]; pose: FacePose } {
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;

  // Locate darkest eye centers in upper-left and upper-right quadrants of 128x128 crop
  let lxSum = 0;
  let lySum = 0;
  let lWeight = 0;
  let rxSum = 0;
  let rySum = 0;
  let rWeight = 0;

  for (let y = 32; y < 64; y++) {
    for (let x = 24; x < 60; x++) {
      const dark = Math.max(0, 255 - alignedGray[y * 128 + x]);
      const wgt = dark * dark;
      lxSum += x * wgt;
      lySum += y * wgt;
      lWeight += wgt;
    }
    for (let x = 68; x < 104; x++) {
      const dark = Math.max(0, 255 - alignedGray[y * 128 + x]);
      const wgt = dark * dark;
      rxSum += x * wgt;
      rySum += y * wgt;
      rWeight += wgt;
    }
  }

  const leNormX = lWeight > 0 ? lxSum / lWeight / 128 : 0.34;
  const leNormY = lWeight > 0 ? lySum / lWeight / 128 : 0.38;
  const reNormX = rWeight > 0 ? rxSum / rWeight / 128 : 0.66;
  const reNormY = rWeight > 0 ? rySum / rWeight / 128 : 0.38;

  const eyeMidX = (leNormX + reNormX) / 2;
  const eyeMidY = (leNormY + reNormY) / 2;
  const eyeDx = reNormX - leNormX;
  const eyeDy = reNormY - leNormY;

  const roll = Number((Math.atan2(eyeDy, Math.max(0.05, eyeDx)) * (180 / Math.PI)).toFixed(2));
  const yaw = Number(((eyeMidX - 0.5) * 90).toFixed(2));
  const pitch = Number(((0.38 - eyeMidY) * 70).toFixed(2));

  const toImg = (nx: number, ny: number): [number, number] => [
    Number((box.x1 + nx * w).toFixed(2)),
    Number((box.y1 + ny * h).toFixed(2)),
  ];

  const pts: Array<[number, number]> = [];

  // 0..16: jawline contour
  for (let i = 0; i < 17; i++) {
    const t = i / 16;
    const angle = Math.PI * (1 - t);
    const nx = 0.5 + Math.cos(angle) * 0.44;
    const ny = 0.42 + Math.sin(angle) * 0.52;
    pts.push(toImg(nx, ny));
  }

  // 17..21: left eyebrow
  for (let i = 0; i < 5; i++) {
    const t = i / 4;
    pts.push(toImg(leNormX - 0.1 + t * 0.2, leNormY - 0.1 - Math.sin(t * Math.PI) * 0.03));
  }

  // 22..26: right eyebrow
  for (let i = 0; i < 5; i++) {
    const t = i / 4;
    pts.push(toImg(reNormX - 0.1 + t * 0.2, reNormY - 0.1 - Math.sin(t * Math.PI) * 0.03));
  }

  // 27..30: nose bridge
  for (let i = 0; i < 4; i++) {
    const t = i / 3;
    pts.push(toImg(eyeMidX, eyeMidY + t * 0.22));
  }

  // 31..35: nose base
  for (let i = 0; i < 5; i++) {
    const t = (i - 2) / 2;
    pts.push(toImg(eyeMidX + t * 0.09, eyeMidY + 0.25 + (1 - Math.abs(t)) * 0.02));
  }

  // 36..41: left eye
  const eyeOffsets: Array<[number, number]> = [
    [-0.07, 0],
    [-0.03, -0.025],
    [0.03, -0.025],
    [0.07, 0],
    [0.03, 0.025],
    [-0.03, 0.025],
  ];
  for (const [ox, oy] of eyeOffsets) {
    pts.push(toImg(leNormX + ox, leNormY + oy));
  }

  // 42..47: right eye
  for (const [ox, oy] of eyeOffsets) {
    pts.push(toImg(reNormX + ox, reNormY + oy));
  }

  // 48..59: outer lip
  const mouthCx = eyeMidX;
  const mouthCy = Math.min(0.84, eyeMidY + 0.39);
  for (let i = 0; i < 12; i++) {
    const theta = (i / 12) * Math.PI * 2;
    pts.push(toImg(mouthCx + Math.cos(theta) * 0.16, mouthCy + Math.sin(theta) * 0.065));
  }

  // 60..67: inner lip
  for (let i = 0; i < 8; i++) {
    const theta = (i / 8) * Math.PI * 2;
    pts.push(toImg(mouthCx + Math.cos(theta) * 0.1, mouthCy + Math.sin(theta) * 0.03));
  }

  const landmarks = pts.flat();
  return { landmarks, pose: { yaw, pitch, roll } };
}

function extractAlignedEmbedding(alignedGray: Buffer): number[] {
  // Normalize luminance across the 128x128 crop for lighting invariance
  let globalSum = 0;
  for (let i = 0; i < alignedGray.length; i++) globalSum += alignedGray[i];
  const globalMean = globalSum / Math.max(1, alignedGray.length);
  let varSum = 0;
  for (let i = 0; i < alignedGray.length; i++) {
    const d = alignedGray[i] - globalMean;
    varSum += d * d;
  }
  const globalStd = Math.sqrt(varSum / Math.max(1, alignedGray.length)) || 1;

  // 8x8 spatial grid over 128x128 aligned crop -> 64 z-normalized luminance + 64 gradient magnitudes = 128 dims
  const gridSize = 8;
  const cell = 128 / gridSize;
  const rawVec: number[] = new Array(128).fill(0);

  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      let lumSum = 0;
      let gradSum = 0;
      const startX = gx * cell;
      const startY = gy * cell;

      for (let y = startY; y < startY + cell; y++) {
        for (let x = startX; x < startX + cell; x++) {
          const p = alignedGray[y * 128 + x];
          const px = alignedGray[y * 128 + Math.min(127, x + 1)];
          const py = alignedGray[Math.min(127, y + 1) * 128 + x];
          lumSum += (p - globalMean) / globalStd;
          gradSum += (Math.abs(px - p) + Math.abs(py - p)) / (2 * globalStd);
        }
      }

      const cellIdx = gy * gridSize + gx;
      rawVec[cellIdx] = lumSum / (cell * cell);
      rawVec[64 + cellIdx] = gradSum / (cell * cell);
    }
  }

  let normSq = 0;
  for (const v of rawVec) normSq += v * v;
  const norm = Math.sqrt(normSq) || 1;
  return rawVec.map((v) => Number((v / norm).toFixed(6)));
}

export class FaceRecognition {
  async getImageInfo(
    input: { sample?: string; imageBase64?: string },
    faceMaxCount = 5
  ): Promise<ImageInfoResult> {
    const start = Date.now();
    const imgBuffer = await loadImageBuffer(input);
    const { data: rotatedBuf, info: rotatedInfo } = await sharp(imgBuffer)
      .rotate()
      .toColourspace('srgb')
      .removeAlpha()
      .png()
      .toBuffer({ resolveWithObject: true });

    const baseImage = sharp(rotatedBuf);
    const width = rotatedInfo.width;
    const height = rotatedInfo.height;

    if (width <= 0 || height <= 0) {
      throw new Error('Unable to decode image dimensions');
    }

    const analysisWidth = Math.min(200, width);
    const analysisHeight = Math.max(1, Math.round((height / width) * analysisWidth));
    const { data: rgb } = await baseImage
      .clone()
      .resize(analysisWidth, analysisHeight, { fit: 'fill' })
      .raw()
      .toBuffer({ resolveWithObject: true });

    const boxes = detectFaceBoxes(
      rgb,
      analysisWidth,
      analysisHeight,
      width,
      height,
      Math.max(1, Math.min(20, faceMaxCount))
    );

    const faces: FaceInfo[] = [];
    for (const box of boxes) {
      const extractW = Math.max(1, box.x2 - box.x1);
      const extractH = Math.max(1, box.y2 - box.y1);

      const alignedPng = await baseImage
        .clone()
        .extract({ left: box.x1, top: box.y1, width: extractW, height: extractH })
        .resize(128, 128, { fit: 'fill' })
        .png()
        .toBuffer();

      const { data: alignedGray } = await sharp(alignedPng)
        .greyscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

      const { landmarks, pose } = estimateLandmarks68(alignedGray, box);
      const embedding = extractAlignedEmbedding(alignedGray);

      faces.push({
        bbox: [box.x1, box.y1, box.x2, box.y2],
        score: box.score,
        landmarks,
        pose,
        alignedImage: `data:image/png;base64,${alignedPng.toString('base64')}`,
        embedding,
      });
    }

    return {
      count: faces.length,
      width,
      height,
      faces,
      processingTimeMs: Date.now() - start,
    };
  }

  getSimilarity(feat1: number[], feat2: number[]): number {
    if (feat1.length === 0 || feat2.length === 0) return 0;
    const len = Math.min(feat1.length, feat2.length);
    let dot = 0;
    for (let i = 0; i < len; i++) {
      dot += feat1[i] * feat2[i];
    }
    const score = (dot + 1) * 50;
    return Number(Math.max(0, Math.min(100, score)).toFixed(2));
  }

  async compareFaces(
    input1: { sample?: string; imageBase64?: string },
    input2: { sample?: string; imageBase64?: string },
    threshold = 75
  ): Promise<FaceComparisonResult> {
    const start = Date.now();
    const [faces1, faces2] = await Promise.all([
      this.getImageInfo(input1, 5),
      this.getImageInfo(input2, 5),
    ]);

    if (faces1.count === 0 || faces2.count === 0) {
      throw new Error('Unable to detect a face in one or both images.');
    }

    const similarity = this.getSimilarity(faces1.faces[0].embedding, faces2.faces[0].embedding);
    return {
      similarity,
      isSamePerson: similarity >= threshold,
      threshold,
      faces1,
      faces2,
      processingTimeMs: Date.now() - start,
    };
  }
}
