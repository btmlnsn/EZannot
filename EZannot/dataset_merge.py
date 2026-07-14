"""Merge annotated image folders into one COCO dataset."""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

EMPTY_COCO = {
	'info': {
		'year': '',
		'version': '1',
		'description': 'EZannot annotations',
		'contributor': '',
		'url': 'https://github.com/yujiahu415/EZannot',
		'date_created': '',
	},
	'licenses': [],
	'categories': [],
	'images': [],
	'annotations': [],
}

COLLISION_KEEP_A = 'keep_a'
COLLISION_KEEP_B = 'keep_b'
COLLISION_REVIEW = 'review'


@dataclass
class MergeOptions:
	collision_policy: str = COLLISION_REVIEW
	# When policy is review: exact file_name -> 'a' or 'b'
	collision_choices: Optional[dict[str, str]] = None
	include_orphans: bool = True


@dataclass
class MergePreview:
	images_a: list[str] = field(default_factory=list)
	images_b: list[str] = field(default_factory=list)
	only_a: list[str] = field(default_factory=list)
	only_b: list[str] = field(default_factory=list)
	both: list[str] = field(default_factory=list)
	annotations_a_count: int = 0
	annotations_b_count: int = 0
	annotated_images_a: int = 0
	annotated_images_b: int = 0
	collision_annotated: list[str] = field(default_factory=list)
	orphans_a: list[str] = field(default_factory=list)
	orphans_b: list[str] = field(default_factory=list)
	categories_a: list[str] = field(default_factory=list)
	categories_b: list[str] = field(default_factory=list)
	has_json_a: bool = False
	has_json_b: bool = False
	byte_diff_collisions: list[str] = field(default_factory=list)

	@property
	def unique_image_count(self) -> int:
		return len(self.only_a) + len(self.only_b) + len(self.both)


@dataclass
class MergeResult:
	images_copied: int
	annotations_written: int
	collisions_resolved: int
	collision_policy: str
	output_path: str
	backed_up_json: bool = False


class MergeError(ValueError):
	pass


def scan_folder_images(path: str) -> list[str]:
	"""Return sorted basenames of supported images in a folder."""
	if not os.path.isdir(path):
		raise MergeError(f'Folder not found: {path}')
	names = []
	for name in os.listdir(path):
		if name.startswith('.'):
			continue
		ext = os.path.splitext(name)[1].lower()
		if ext in IMAGE_EXTENSIONS and os.path.isfile(os.path.join(path, name)):
			names.append(name)
	return sorted(names)


def load_coco(path: str) -> dict:
	"""Load annotations.json from path, or return an empty COCO structure."""
	json_path = os.path.join(path, 'annotations.json') if os.path.isdir(path) else path
	if not os.path.isfile(json_path):
		return deepcopy(EMPTY_COCO)
	try:
		with open(json_path, 'r') as f:
			data = json.load(f)
	except json.JSONDecodeError as exc:
		raise MergeError(f'Invalid JSON in {json_path}: {exc}') from exc
	for key in ('categories', 'images', 'annotations'):
		if key not in data or not isinstance(data[key], list):
			raise MergeError(f'Missing or invalid "{key}" in {json_path}')
	return data


def _folder_summary(folder: str) -> dict:
	images = scan_folder_images(folder)
	json_path = os.path.join(folder, 'annotations.json')
	has_json = os.path.isfile(json_path)
	coco = load_coco(folder) if has_json else deepcopy(EMPTY_COCO)
	image_set = set(images)
	file_to_id = {img['file_name']: img['id'] for img in coco['images']}
	annotated_names = set()
	for ann in coco['annotations']:
		for name, img_id in file_to_id.items():
			if img_id == ann['image_id']:
				annotated_names.add(name)
				break
	orphans = sorted(name for name in file_to_id if name not in image_set)
	categories = [c['name'] for c in coco['categories'] if c.get('id', 0) > 0]
	return {
		'images': images,
		'has_json': has_json,
		'coco': coco,
		'annotated_names': annotated_names,
		'orphans': orphans,
		'categories': categories,
		'annotation_count': len(coco['annotations']),
	}


def summarize_folder(folder: str) -> str:
	"""Short human-readable summary for UI after folder selection."""
	info = _folder_summary(folder)
	n = len(info['images'])
	if not info['has_json']:
		return f'{n} images, no annotations.json (images only)'
	annotated = len(info['annotated_names'] - set(info['orphans']))
	orphan_n = len(info['orphans'])
	parts = [f'{n} images, annotations.json found ({annotated} annotated']
	if orphan_n:
		parts.append(f'{orphan_n} orphans')
	return ', '.join(parts) + ')'


def preview_merge(folder_a: str, folder_b: str) -> MergePreview:
	info_a = _folder_summary(folder_a)
	info_b = _folder_summary(folder_b)
	set_a = set(info_a['images'])
	set_b = set(info_b['images'])
	both = sorted(set_a & set_b)
	collision_annotated = [
		name for name in both
		if name in info_a['annotated_names'] and name in info_b['annotated_names']
	]
	byte_diff = []
	for name in both:
		path_a = os.path.join(folder_a, name)
		path_b = os.path.join(folder_b, name)
		try:
			if os.path.getsize(path_a) != os.path.getsize(path_b):
				byte_diff.append(name)
		except OSError:
			byte_diff.append(name)
	return MergePreview(
		images_a=info_a['images'],
		images_b=info_b['images'],
		only_a=sorted(set_a - set_b),
		only_b=sorted(set_b - set_a),
		both=both,
		annotations_a_count=info_a['annotation_count'],
		annotations_b_count=info_b['annotation_count'],
		annotated_images_a=len(info_a['annotated_names']),
		annotated_images_b=len(info_b['annotated_names']),
		collision_annotated=collision_annotated,
		orphans_a=info_a['orphans'],
		orphans_b=info_b['orphans'],
		categories_a=info_a['categories'],
		categories_b=info_b['categories'],
		has_json_a=info_a['has_json'],
		has_json_b=info_b['has_json'],
		byte_diff_collisions=byte_diff,
	)


def format_preview(preview: MergePreview, folder_out: Optional[str] = None) -> str:
	lines = [
		f'Combined unique images: {preview.unique_image_count}',
		f'  - Only in A: {len(preview.only_a)}',
		f'  - Only in B: {len(preview.only_b)}',
		f'  - In both (name collision): {len(preview.both)}',
		'',
		'Annotations to merge:',
		f'  - From A: {preview.annotations_a_count} objects on {preview.annotated_images_a} images',
		f'  - From B: {preview.annotations_b_count} objects on {preview.annotated_images_b} images',
		f'  - Collisions (both have annotations): {len(preview.collision_annotated)} images',
	]
	if preview.orphans_a or preview.orphans_b:
		lines.append('')
		if preview.orphans_a:
			lines.append(f'Dataset A orphans: {len(preview.orphans_a)}')
		if preview.orphans_b:
			lines.append(f'Dataset B orphans: {len(preview.orphans_b)}')
	if preview.byte_diff_collisions:
		lines.append('')
		lines.append(
			f'Warning: {len(preview.byte_diff_collisions)} colliding file(s) differ in size.'
		)
	if not preview.has_json_a and not preview.has_json_b:
		lines.append('')
		lines.append('No annotations to merge; copying images only.')
	lines.append('')
	if folder_out:
		lines.append(f'Output folder: {folder_out}')
	else:
		lines.append('Output folder: not selected yet')
	return '\n'.join(lines)


def _validate_folders(folder_a: str, folder_b: str, folder_out: str) -> None:
	for label, path in (('Dataset A', folder_a), ('Dataset B', folder_b), ('Output', folder_out)):
		if not path or not os.path.isdir(path):
			raise MergeError(f'{label} folder is missing or invalid.')
	real_a = os.path.realpath(folder_a)
	real_b = os.path.realpath(folder_b)
	real_out = os.path.realpath(folder_out)
	if real_a == real_b:
		raise MergeError('Select two different folders.')
	if real_out in (real_a, real_b):
		raise MergeError('Output must be a separate folder.')
	if not os.access(folder_out, os.W_OK):
		raise MergeError(f'Output folder is not writable: {folder_out}')


def _side_for_collision(name: str, policy: str, choices: Optional[dict[str, str]]) -> str:
	if policy == COLLISION_KEEP_A:
		return 'a'
	if policy == COLLISION_KEEP_B:
		return 'b'
	if policy == COLLISION_REVIEW:
		side = (choices or {}).get(name)
		if side not in ('a', 'b'):
			raise MergeError(f'No A/B choice recorded for colliding file: {name}')
		return side
	raise MergeError(f'Unsupported collision policy: {policy}')


def _image_size(path: str, fallback: Optional[dict] = None) -> tuple[int, int]:
	if fallback and fallback.get('width') and fallback.get('height'):
		return int(fallback['width']), int(fallback['height'])
	with Image.open(path) as im:
		return im.size


def _anns_for_image(coco: dict, image_id) -> list[dict]:
	return [ann for ann in coco['annotations'] if ann['image_id'] == image_id]


def _build_file_index(coco: dict) -> dict[str, dict]:
	return {img['file_name']: img for img in coco['images']}


def _union_categories(coco_a: dict, coco_b: dict) -> tuple[list[dict], dict, dict]:
	"""Return merged categories and maps from old category_id -> new id for A and B."""
	ordered_names: list[str] = []
	for coco in (coco_a, coco_b):
		for cat in coco['categories']:
			if cat.get('id', 0) <= 0:
				continue
			name = cat['name']
			if name not in ordered_names:
				ordered_names.append(name)
	categories = [
		{'id': i + 1, 'name': name, 'supercategory': 'none'}
		for i, name in enumerate(ordered_names)
	]
	name_to_new = {c['name']: c['id'] for c in categories}

	def map_for(coco: dict) -> dict:
		mapping = {}
		for cat in coco['categories']:
			if cat.get('id', 0) <= 0:
				continue
			mapping[cat['id']] = name_to_new[cat['name']]
		return mapping

	return categories, map_for(coco_a), map_for(coco_b)


def _resolve_entries(
	preview: MergePreview,
	folder_a: str,
	folder_b: str,
	index_a: dict,
	index_b: dict,
	policy: str,
	include_orphans: bool,
	collision_choices: Optional[dict[str, str]] = None,
) -> list[dict]:
	"""Build output image list: out_name, src_folder, src_name, side, img_info."""
	final_entries: list[dict] = []
	taken_names: set[str] = set()

	def add_entry(out_name, src_folder, src_name, side, img_info):
		taken_names.add(out_name)
		final_entries.append({
			'out_name': out_name,
			'src_folder': src_folder,
			'src_name': src_name,
			'side': side,
			'img_info': img_info,
		})

	for name in preview.only_a:
		add_entry(name, folder_a, name, 'a', index_a.get(name))
	for name in preview.only_b:
		add_entry(name, folder_b, name, 'b', index_b.get(name))

	for name in preview.both:
		side = _side_for_collision(name, policy, collision_choices)
		if side == 'a':
			add_entry(name, folder_a, name, 'a', index_a.get(name))
		else:
			add_entry(name, folder_b, name, 'b', index_b.get(name))

	if not include_orphans:
		return final_entries

	orphans_a = set(preview.orphans_a)
	orphans_b = set(preview.orphans_b)

	for name in preview.orphans_a:
		if name in taken_names:
			continue
		if name in orphans_b:
			if _side_for_collision(name, policy, collision_choices) == 'b':
				continue
		add_entry(name, folder_a, name, 'a', index_a.get(name))

	for name in preview.orphans_b:
		if name in orphans_a:
			side = _side_for_collision(name, policy, collision_choices)
			if side == 'a':
				continue
			final_entries[:] = [e for e in final_entries if e['out_name'] != name]
			taken_names.discard(name)
			add_entry(name, folder_b, name, 'b', index_b.get(name))
			continue
		if name in taken_names:
			continue
		add_entry(name, folder_b, name, 'b', index_b.get(name))

	return final_entries


def merge_datasets(
	folder_a: str,
	folder_b: str,
	folder_out: str,
	options: Optional[MergeOptions] = None,
	progress_callback: Optional[Callable[[str], None]] = None,
) -> MergeResult:
	"""Copy images and write a merged annotations.json into folder_out."""
	options = options or MergeOptions()
	policy = options.collision_policy
	if policy not in (COLLISION_KEEP_A, COLLISION_KEEP_B, COLLISION_REVIEW):
		raise MergeError(f'Unsupported collision policy: {policy}')

	_validate_folders(folder_a, folder_b, folder_out)

	preview = preview_merge(folder_a, folder_b)
	if preview.unique_image_count == 0 and not (preview.orphans_a or preview.orphans_b):
		raise MergeError('No supported images found.')

	if policy == COLLISION_REVIEW and preview.both:
		missing = [n for n in preview.both if (options.collision_choices or {}).get(n) not in ('a', 'b')]
		if missing:
			raise MergeError(
				f'Collision review incomplete ({len(missing)} file(s) without an A/B choice).'
			)

	coco_a = load_coco(folder_a)
	coco_b = load_coco(folder_b)
	index_a = _build_file_index(coco_a)
	index_b = _build_file_index(coco_b)
	categories, cat_map_a, cat_map_b = _union_categories(coco_a, coco_b)
	final_entries = _resolve_entries(
		preview,
		folder_a,
		folder_b,
		index_a,
		index_b,
		policy,
		options.include_orphans,
		options.collision_choices,
	)

	def report(msg: str):
		if progress_callback:
			progress_callback(msg)

	report('Copying images…')
	copied = 0
	for entry in final_entries:
		src = os.path.join(entry['src_folder'], entry['src_name'])
		dst = os.path.join(folder_out, entry['out_name'])
		if not os.path.isfile(src):
			continue
		try:
			shutil.copy2(src, dst)
		except OSError as exc:
			raise MergeError(
				f'Failed copying {src} → {dst}: {exc}. '
				'Partial image copies may remain; annotations.json was not written.'
			) from exc
		copied += 1

	report('Merging annotations…')
	merged = deepcopy(EMPTY_COCO)
	merged['categories'] = categories
	annotation_id = 0

	for new_image_id, entry in enumerate(final_entries):
		src_path = os.path.join(entry['src_folder'], entry['src_name'])
		img_info = entry['img_info']
		if os.path.isfile(src_path):
			width, height = _image_size(src_path, img_info)
		elif img_info:
			width = int(img_info.get('width') or 0)
			height = int(img_info.get('height') or 0)
		else:
			width, height = 0, 0

		merged['images'].append({
			'id': new_image_id,
			'file_name': entry['out_name'],
			'width': width,
			'height': height,
		})

		if entry['side'] == 'a' and img_info is not None:
			src_anns = _anns_for_image(coco_a, img_info['id'])
			cat_map = cat_map_a
		elif entry['side'] == 'b' and img_info is not None:
			src_anns = _anns_for_image(coco_b, img_info['id'])
			cat_map = cat_map_b
		else:
			src_anns = []
			cat_map = {}

		for ann in src_anns:
			new_ann = deepcopy(ann)
			new_ann['id'] = annotation_id
			new_ann['image_id'] = new_image_id
			old_cat = ann.get('category_id')
			if old_cat in cat_map:
				new_ann['category_id'] = cat_map[old_cat]
			merged['annotations'].append(new_ann)
			annotation_id += 1

	report('Writing annotations.json…')
	out_json = os.path.join(folder_out, 'annotations.json')
	backed_up = False
	if os.path.isfile(out_json):
		shutil.copy2(out_json, out_json + '.bak')
		backed_up = True

	tmp_json = out_json + '.tmp'
	with open(tmp_json, 'w') as f:
		json.dump(merged, f)
	os.replace(tmp_json, out_json)

	return MergeResult(
		images_copied=copied,
		annotations_written=len(merged['annotations']),
		collisions_resolved=len(preview.both),
		collision_policy=policy,
		output_path=folder_out,
		backed_up_json=backed_up,
	)


def policy_label(policy: str) -> str:
	return {
		COLLISION_KEEP_A: 'Keep Dataset A for all',
		COLLISION_KEEP_B: 'Keep Dataset B for all',
		COLLISION_REVIEW: 'Review each collision',
	}.get(policy, policy)


def policy_description(policy: str) -> str:
	return {
		COLLISION_KEEP_A: 'Keep Dataset A for all collisions',
		COLLISION_KEEP_B: 'Keep Dataset B for all collisions',
		COLLISION_REVIEW: 'Review each collision and choose A (left) or B (right)',
	}.get(policy, policy)

