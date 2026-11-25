import express, { NextFunction, Request, Response } from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import { FaceRecognition } from './src/faceSdk.js';

const PORT = 3000;
const HOST = '0.0.0.0';
const parsedEnvThreshold = Number(process.env.FACE_SIMILARITY_THRESHOLD);
const DEFAULT_THRESHOLD =
  Number.isFinite(parsedEnvThreshold) && parsedEnvThreshold >= 0 && parsedEnvThreshold <= 100
    ? parsedEnvThreshold
    : 75;

const app = express();
const faceSdk = new FaceRecognition();

interface ComparisonRecord {
  id: string;
  label1: string;
  label2: string;
  similarity: number;
  threshold: number;
  isSamePerson: boolean;
  createdAt: string;
}

const comparisonHistory = new Map<string, ComparisonRecord>();

app.use(express.json({ limit: '15mb' }));
app.use(express.static(path.resolve(process.cwd(), 'public')));

app.get('/api/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    sdk: 'Open Source Face Recognition SDK',
    defaultThreshold: DEFAULT_THRESHOLD,
  });
});

app.get('/api/samples', async (_req: Request, res: Response) => {
  try {
    const samplesDir = path.resolve(process.cwd(), 'public', 'samples');
    const files = await fs.readdir(samplesDir);
    const samples = files
      .filter((f) => /\.(jpg|jpeg|png|webp|bmp)$/i.test(f))
      .sort()
      .map((name) => ({
        name,
        url: `/samples/${name}`,
      }));
    res.json({ samples });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

app.post('/api/face/info', async (req: Request, res: Response) => {
  try {
    const body = (req.body ?? {}) as {
      sample?: string;
      imageBase64?: string;
      faceMaxCount?: number;
    };
    const faceMaxCount = typeof body.faceMaxCount === 'number' ? body.faceMaxCount : 5;
    const result = await faceSdk.getImageInfo(
      { sample: body.sample, imageBase64: body.imageBase64 },
      faceMaxCount
    );
    res.json(result);
  } catch (err) {
    res.status(400).json({ error: (err as Error).message });
  }
});

app.post('/api/face/compare', async (req: Request, res: Response) => {
  try {
    const body = (req.body ?? {}) as {
      image1?: { sample?: string; imageBase64?: string; label?: string };
      image2?: { sample?: string; imageBase64?: string; label?: string };
      threshold?: number;
    };

    const input1 = body.image1 ?? { sample: '1.jpg', label: '1.jpg' };
    const input2 = body.image2 ?? { sample: '2.png', label: '2.png' };
    const rawThreshold =
      typeof body.threshold === 'number' && Number.isFinite(body.threshold)
        ? body.threshold
        : DEFAULT_THRESHOLD;
    const threshold = Math.max(0, Math.min(100, rawThreshold));

    const result = await faceSdk.compareFaces(input1, input2, threshold);

    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const record: ComparisonRecord = {
      id,
      label1: input1.label ?? input1.sample ?? 'Uploaded Image 1',
      label2: input2.label ?? input2.sample ?? 'Uploaded Image 2',
      similarity: result.similarity,
      threshold: result.threshold,
      isSamePerson: result.isSamePerson,
      createdAt: new Date().toISOString(),
    };
    comparisonHistory.set(id, record);
    if (comparisonHistory.size > 25) {
      const oldestKey = comparisonHistory.keys().next().value;
      if (oldestKey) comparisonHistory.delete(oldestKey);
    }

    res.json(result);
  } catch (err) {
    res.status(400).json({ error: (err as Error).message });
  }
});

app.get('/api/history', (_req: Request, res: Response) => {
  res.json({
    records: Array.from(comparisonHistory.values()).reverse(),
  });
});

app.use((_req: Request, res: Response) => {
  res.sendFile(path.resolve(process.cwd(), 'public', 'index.html'));
});

app.use((err: Error & { status?: number }, _req: Request, res: Response, _next: NextFunction) => {
  void _next;
  const status = typeof err.status === 'number' && err.status >= 400 ? err.status : 500;
  res.status(status).json({ error: err.message || 'Internal server error' });
});

app.listen(PORT, HOST, () => {
  console.log(`Face Recognition SDK server listening on http://${HOST}:${PORT}`);
});
