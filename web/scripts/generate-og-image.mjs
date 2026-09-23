import sharp from 'sharp';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const svg = join(here, '..', 'public', 'og-image.svg');
const png = join(here, '..', 'public', 'og-image.png');

await sharp(svg, { density: 96 })
  .resize(1200, 630)
  .png()
  .toFile(png);

console.log(`[og-image] regenerated ${png}`);
