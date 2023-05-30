import json
from collections import defaultdict

import pandas as pd
import translators as ts

train_data = pd.read_parquet('../../data/hackathon_files_for_participants_ozon/train_data.parquet')
colors = set()
for color in train_data['color_parsed']:
    if color is not None and color.any():
        colors.update(color)

matching = defaultdict(list)
for color1 in colors:
    for color2 in colors:
        if color1 != color2 and '-' not in color2 and color2.startswith(color1):
            matching[color1].append(color2)
    if color1.isascii():
        translation = ts.translate_text(color1, to_language='ru').lower().replace('ё', 'е')
        if translation in colors:
            matching[color1].append(translation)

matching_update = {
    'бел': ['белый'],
    'голуб': ['синий'],
    'голубой': ['синий'],
    'крас': ['красный'],
    'сер': ['серый'],
    'фиол': ['фиолетовый'],
    'чер': ['черный'],
    'emerald': ['изумрудный'],
    }
matching.update(matching_update)
matching = {k: v[0] for k, v in matching.items()}

with open('../files/colors_mapping.json', 'w', encoding='utf-8') as out:
    json.dump(matching, out, ensure_ascii=False, indent=1, sort_keys=True)

colors = {color for color in colors if color not in matching}
colors.update({'красно-коричневый', 'бледно-пурпурный', 'коричнево-бежевый'})
with open('../files/colors.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(sorted(colors)))
