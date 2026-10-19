"""Compose evidence-based localization examples from saved simulator results."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = {r['query']: r for r in json.loads((args.dataset / 'results/localization.json').read_text())}
frames = {r['id']: r for r in json.loads((args.dataset / 'manifest.json').read_text())}
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
def font(n):
    return ImageFont.truetype(font_path, n)

cards = []
for index, query in enumerate(['query_03', 'query_09', 'query_19'], 1):
    row = rows[query]
    est = row['estimate']
    ref = est['reference']
    failure = row['position_error_m'] > 1
    accent = '#ff8c80' if failure else '#64ddbc'
    card = Image.new('RGB', (1344, 1190), '#101b2b')
    draw = ImageDraw.Draw(card)
    title = 'Failure: repeated facade windows' if failure else 'Successful localization'
    draw.text((32, 24), f'{index:02d}  {title}', font=font(36), fill=accent)
    draw.text((32, 73), 'Actual Webots simulator frames | DINOv2 retrieval + SIFT / RGB-D PnP', font=font(23), fill='#c5d1e2')
    draw.text((32, 120), f'Mapped reference: {ref}', font=font(25), fill='white')
    draw.text((672, 120), f'Held-out camera view: {query}', font=font(25), fill='white')
    for name, x in [(ref, 32), (query, 672)]:
        with Image.open(args.dataset / frames[name]['image']) as source:
            card.paste(source.convert('RGB').resize((640, 384)), (x, 160))
    draw.text((32, 565), 'Geometric matches accepted by RANSAC (not guaranteed correct)', font=font(25), fill='white')
    with Image.open(args.dataset / 'results' / f'{query}_matches.jpg') as matches:
        card.paste(matches.convert('RGB').resize((1280, 384)), (32, 608))
    metrics = (f"3D position error: {row['position_error_m']:.2f} m"
               f"     Inliers: {est['inliers']}/{est['matches']}"
               f"     Reprojection: {est['median_reprojection_px']:.2f} px")
    draw.text((32, 1018), metrics, font=font(27), fill=accent)
    note = ('Repeated windows support a wrong pose even with a tiny reprojection error.'
            if failure else 'The estimated camera position agrees closely with simulator ground truth.')
    draw.text((32, 1065), note, font=font(24), fill='white')
    draw.text((32, 1120), 'Offline mapped-scene experiment; not real-world validation or autonomous navigation.', font=font(22), fill='#a5b6ca')
    path = args.output / f'{query}_example.png'
    card.save(path)
    with Image.open(path) as check:
        check.verify()
    cards.append(card)
    print(path)

overview = Image.new('RGB', (1344, 1190 * len(cards)), '#101b2b')
for index, card in enumerate(cards):
    overview.paste(card, (0, index * 1190))
overview.save(args.output / 'localization_examples.jpg', quality=92)
