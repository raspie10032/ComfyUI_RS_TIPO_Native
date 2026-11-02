"""Make an image comparison from an existing image-test results.json."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    cases = list(dict.fromkeys(row['case'] for row in rows))
    modes = ('original', 'previous', 'focused')
    cell_w, cell_h = 384, 332
    sheet = Image.new('RGB', (cell_w * 3, 44 + cell_h * len(cases)), '#20242b')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for x, title in enumerate(('Source prompt', 'Previous expansion', 'Candidate expansion')):
        draw.text((x * cell_w + 12, 12), title, font=font, fill='white')
    for y, case in enumerate(cases):
        for x, mode in enumerate(modes):
            row = next((r for r in rows if r['case'] == case and r['mode'] == mode), None)
            if row is None:
                continue
            tile = Image.open(row['file']).convert('RGB')
            tile.thumbnail((cell_w, cell_h - 32), Image.Resampling.LANCZOS)
            origin = (x * cell_w, 44 + y * cell_h)
            sheet.paste(tile, origin)
            draw.text((origin[0] + 12, origin[1] + cell_h - 27), case, font=font, fill='white')
    sheet.save(args.output)


if __name__ == '__main__':
    main()
