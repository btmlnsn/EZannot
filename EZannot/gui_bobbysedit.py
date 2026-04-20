import os
import cv2
import wx
import json
import random
import torch
import copy
import numpy as np
from collections import deque
from pathlib import Path
from PIL import Image
from screeninfo import get_monitors
from EZannot.sam2.build_sam import build_sam2
from EZannot.sam2.sam2_image_predictor import SAM2ImagePredictor
from .gui_annotating import ColorPicker
from .tools import read_annotation, mask_to_polygon, generate_annotation


the_absolute_current_path = str(Path(__file__).resolve().parent)


class PanelLv2_BobbysEdit(wx.Panel):

	def __init__(self, parent):

		super().__init__(parent)
		self.notebook = parent
		self.path_to_images = None
		self.result_path = None
		self.model_cp = None
		self.model_cfg = None
		self.color_map = {}
		self.aug_methods = []

		self.display_window()


	def display_window(self):

		panel = self
		boxsizer = wx.BoxSizer(wx.VERTICAL)

		module_input = wx.BoxSizer(wx.HORIZONTAL)
		button_input = wx.Button(panel, label='Select the image(s)\nto annotate', size=(300, 40))
		button_input.Bind(wx.EVT_BUTTON, self.select_images)
		wx.Button.SetToolTip(button_input, 'Select one or more images. Common image formats (jpg, png, tif) are supported. If there is an annotation file in the same folder, EZannot will read the annotation file and show all the existing annotations.')
		self.text_input = wx.StaticText(panel, label='None.', style=wx.ALIGN_LEFT | wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		module_input.Add(self.text_input, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(0, 10, 0)
		boxsizer.Add(module_input, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(0, 5, 0)

		module_outputfolder = wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder = wx.Button(panel, label='Select a folder to store\nthe annotated images', size=(300, 40))
		button_outputfolder.Bind(wx.EVT_BUTTON, self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder, 'Copies of images (including augmented ones) and the annotation file will be stored in this folder. The annotation file for the original (unaugmented) images will be stored in the origianl image folder.')
		self.text_outputfolder = wx.StaticText(panel, label='None.', style=wx.ALIGN_LEFT | wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		module_outputfolder.Add(self.text_outputfolder, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(module_outputfolder, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(0, 5, 0)

		module_model = wx.BoxSizer(wx.HORIZONTAL)
		button_model = wx.Button(panel, label='Set up the SAM2 model for\nAI-assisted annotation', size=(300, 40))
		button_model.Bind(wx.EVT_BUTTON, self.select_model)
		wx.Button.SetToolTip(button_model, 'Choose the SAM2 model. If select from a folder, make sure the folder stores a checkpoint (*.pt) file and a corresponding model config (*.yaml) file.')
		self.text_model = wx.StaticText(panel, label='None.', style=wx.ALIGN_LEFT | wx.ST_ELLIPSIZE_END)
		module_model.Add(button_model, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		module_model.Add(self.text_model, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(module_model, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(0, 5, 0)

		module_classes = wx.BoxSizer(wx.HORIZONTAL)
		button_classes = wx.Button(panel, label='Specify the object classes and\ntheir annotation colors', size=(300, 40))
		button_classes.Bind(wx.EVT_BUTTON, self.specify_classes)
		wx.Button.SetToolTip(button_classes, 'Enter the name of each class and specify its annotation color.')
		self.text_classes = wx.StaticText(panel, label='None.', style=wx.ALIGN_LEFT | wx.ST_ELLIPSIZE_END)
		module_classes.Add(button_classes, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		module_classes.Add(self.text_classes, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(module_classes, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(0, 5, 0)

		module_augmentation = wx.BoxSizer(wx.HORIZONTAL)
		button_augmentation = wx.Button(panel, label='Specify the augmentation methods\nfor the annotated images', size=(300, 40))
		button_augmentation.Bind(wx.EVT_BUTTON, self.specify_augmentation)
		wx.Button.SetToolTip(button_augmentation,
			'Augmentation can greatly enhance the training efficiency. But for the first time of annotating an image set, you can skip this to keep an unaugmented, origianl annotated image set and import it to EZannot later to perform augmentation.')
		self.text_augmentation = wx.StaticText(panel, label='None.', style=wx.ALIGN_LEFT | wx.ST_ELLIPSIZE_END)
		module_augmentation.Add(button_augmentation, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		module_augmentation.Add(self.text_augmentation, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(module_augmentation, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(0, 5, 0)

		button_startannotation = wx.Button(panel, label="Start Bobby's Interactive Edit", size=(300, 40))
		button_startannotation.Bind(wx.EVT_BUTTON, self.start_annotation)
		wx.Button.SetToolTip(button_startannotation, "Start Bobby's Interactive Edit mode.")
		boxsizer.Add(0, 5, 0)
		boxsizer.Add(button_startannotation, 0, wx.RIGHT | wx.ALIGN_RIGHT, 90)
		boxsizer.Add(0, 10, 0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self, event):

		valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
		dialog = wx.DirDialog(self, 'Select an image folder', '', style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal() == wx.ID_OK:
			selected_folder = dialog.GetPath()
			image_paths = []
			for file_name in sorted(os.listdir(selected_folder)):
				file_path = os.path.join(selected_folder, file_name)
				if os.path.isfile(file_path) and file_name.lower().endswith(valid_extensions):
					image_paths.append(file_path)
			if len(image_paths) == 0:
				wx.MessageDialog(
					self,
					'No valid images were found in the selected folder.\nSupported formats: .jpg, .jpeg, .png, .tif, .tiff, .bmp',
					'No images found',
					wx.OK | wx.ICON_WARNING
				).ShowModal()
				dialog.Destroy()
				return
			self.path_to_images = image_paths
			self.text_input.SetLabel('📁 ' + selected_folder + ' — ' + str(len(self.path_to_images)) + ' images found')
		dialog.Destroy()


	def select_outpath(self, event):

		dialog = wx.DirDialog(self, 'Select a directory', '', style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal() == wx.ID_OK:
			self.result_path = dialog.GetPath()
			self.text_outputfolder.SetLabel('The annotated images will be in: ' + self.result_path + '.')
		dialog.Destroy()


	def select_model(self, event):

		path_to_sam2_model = None
		sam2_model_path = os.path.join(the_absolute_current_path, 'sam2 models')
		sam2_models = [i for i in os.listdir(sam2_model_path) if os.path.isdir(os.path.join(sam2_model_path, i))]
		if '__pycache__' in sam2_models:
			sam2_models.remove('__pycache__')
		if '__init__' in sam2_models:
			sam2_models.remove('__init__')
		if '__init__.py' in sam2_models:
			sam2_models.remove('__init__.py')
		sam2_models.sort()
		if 'Choose a new directory of the SAM2 model' not in sam2_models:
			sam2_models.append('Choose a new directory of the SAM2 model')

		dialog = wx.SingleChoiceDialog(self, message='Select a SAM2 model for AI-assisted annotation.', caption='Select a SAM2 model', choices=sam2_models)
		if dialog.ShowModal() == wx.ID_OK:
			sam2_model = dialog.GetStringSelection()
			if sam2_model == 'Choose a new directory of the SAM2 model':
				dialog1 = wx.DirDialog(self, 'Select a directory', '', style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal() == wx.ID_OK:
					path_to_sam2_model = dialog1.GetPath()
				else:
					path_to_sam2_model = None
				dialog1.Destroy()
			else:
				path_to_sam2_model = os.path.join(sam2_model_path, sam2_model)
		dialog.Destroy()

		if path_to_sam2_model is None:
			wx.MessageBox('No SAM2 model is set up. The AI assistance function is OFF.', 'AI assistance OFF', wx.ICON_INFORMATION)
			self.text_model.SetLabel('No SAM2 model is set up. The AI assistance function is OFF.')
		else:
			self.model_cp = None
			self.model_cfg = None
			for i in os.listdir(path_to_sam2_model):
				if i.endswith('.pt') and i.split('sam')[0] != '._':
					self.model_cp = os.path.join(path_to_sam2_model, i)
				if i.endswith('.yaml') and i.split('sam')[0] != '._':
					self.model_cfg = os.path.join(path_to_sam2_model, i)
			if self.model_cp is None:
				self.text_model.SetLabel('Missing checkpoint file.')
			elif self.model_cfg is None:
				self.text_model.SetLabel('Missing config file.')
			else:
				self.text_model.SetLabel('Checkpoint: ' + str(os.path.basename(self.model_cp)) + '; Config: ' + str(os.path.basename(self.model_cfg)) + '.')


	def specify_classes(self, event):

		if self.path_to_images is None:

			wx.MessageBox('No input images(s).', 'Error', wx.OK | wx.ICON_ERROR)

		else:

			annotation_files = []
			color_map = {}
			self.color_map = {}
			classnames = ''
			entry = None
			for i in os.listdir(os.path.dirname(self.path_to_images[0])):
				if i.endswith('.json'):
					annotation_files.append(os.path.join(os.path.dirname(self.path_to_images[0]), i))

			if len(annotation_files) > 0:
				for annotation_file in annotation_files:
					if os.path.exists(annotation_file):
						annotation = json.load(open(annotation_file))
						for i in annotation['categories']:
							if i['id'] > 0:
								classname = i['name']
								if classname not in classnames:
									classnames = classnames + classname + ','
				classnames = classnames[:-1]
				dialog = wx.MessageDialog(self, 'Current classnames are: ' + classnames + '.\nDo you want to modify the classnames?', 'Modify classnames?', wx.YES_NO | wx.ICON_QUESTION)
				if dialog.ShowModal() == wx.ID_YES:
					dialog1 = wx.TextEntryDialog(self, 'Enter the names of objects to annotate\n(use "," to separate each name)', 'Object class names', value=classnames)
					if dialog1.ShowModal() == wx.ID_OK:
						entry = dialog1.GetValue()
					dialog1.Destroy()
				else:
					entry = classnames
				dialog.Destroy()
			else:
				dialog = wx.TextEntryDialog(self, 'Enter the names of objects to annotate\n(use "," to separate each name)', 'Object class names')
				if dialog.ShowModal() == wx.ID_OK:
					entry = dialog.GetValue()
				dialog.Destroy()

			if entry:
				try:
					for i in entry.split(','):
						color_map[i] = '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
				except:
					color_map = {}
					wx.MessageBox('Please enter the object class names in\ncorrect format! For example: apple,orange,pear', 'Error', wx.OK | wx.ICON_ERROR)

			if len(color_map) > 0:
				for classname in color_map:
					dialog = ColorPicker(self, str(classname), [classname, color_map[classname]])
					if dialog.ShowModal() == wx.ID_OK:
						(r, b, g, _) = dialog.color_picker.GetColour()
						self.color_map[classname] = (r, b, g)
					else:
						self.color_map[classname] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
					dialog.Destroy()
				self.text_classes.SetLabel('Classname:color: ' + str(self.color_map) + '.')
			else:
				self.text_classes.SetLabel('None.')


	def specify_augmentation(self, event):

		aug_methods = ['random rotation', 'horizontal flipping', 'vertical flipping', 'random brightening', 'random dimming', 'random blurring']
		selected = ''
		dialog = wx.MultiChoiceDialog(self, message='Data augmentation methods', caption='Augmentation methods', choices=aug_methods)
		if dialog.ShowModal() == wx.ID_OK:
			self.aug_methods = [aug_methods[i] for i in dialog.GetSelections()]
			for i in self.aug_methods:
				if selected == '':
					selected = selected + i
				else:
					selected = selected + ',' + i
		else:
			self.aug_methods = []
			selected = 'none'
		dialog.Destroy()

		if len(self.aug_methods) <= 0:
			selected = 'none'

		self.text_augmentation.SetLabel('Augmentation methods: ' + selected + '.')


	def start_annotation(self, event):

		if self.path_to_images is None or self.result_path is None or len(self.color_map) == 0:
			wx.MessageBox('No input images(s) / output folder / class names.', 'Error', wx.OK | wx.ICON_ERROR)
		else:
			WindowLv3_BobbysInteractiveEdit(
				None,
				"Bobby's Interactive Edit",
				self.path_to_images,
				self.result_path,
				self.color_map,
				self.aug_methods,
				model_cp=self.model_cp,
				model_cfg=self.model_cfg,
			)


class ThumbnailGridPopup(wx.PopupTransientWindow):

	def __init__(self, parent_frame, indices, image_paths, navigate_callback, thumb_size=80, columns=4, max_height=400):

		super().__init__(parent_frame, style=wx.BORDER_SIMPLE)
		self.parent_frame = parent_frame
		self.indices = indices
		self.image_paths = image_paths
		self.navigate_callback = navigate_callback
		self.thumb_size = thumb_size
		self.columns = max(1, columns)
		self.max_height = max_height
		self.thumbnail_bitmaps = []

		root_panel = wx.Panel(self)
		root_sizer = wx.BoxSizer(wx.VERTICAL)
		scrolled = wx.ScrolledWindow(root_panel, style=wx.VSCROLL)
		scrolled.SetScrollRate(8, 8)
		content_panel = wx.Panel(scrolled)
		content_sizer = wx.BoxSizer(wx.VERTICAL)

		if len(self.indices) == 0:
			none_label = wx.StaticText(content_panel, label='(none)')
			none_label.SetForegroundColour(wx.Colour(120, 120, 120))
			content_sizer.AddStretchSpacer(1)
			content_sizer.Add(none_label, 0, wx.ALIGN_CENTER | wx.ALL, 20)
			content_sizer.AddStretchSpacer(1)
		else:
			grid = wx.FlexGridSizer(cols=self.columns, hgap=10, vgap=10)
			for image_idx in self.indices:
				cell = wx.Panel(content_panel)
				cell_sizer = wx.BoxSizer(wx.VERTICAL)
				image_path = self.image_paths[image_idx]
				file_name = os.path.basename(image_path)
				bitmap = self.parent_frame.create_thumbnail_bitmap(image_path, max_size=self.thumb_size)
				self.thumbnail_bitmaps.append(bitmap)

				bmp_ctrl = wx.StaticBitmap(cell, bitmap=bitmap)
				label = wx.StaticText(cell, label=file_name)
				label.SetMinSize((self.thumb_size + 8, -1))
				label.Wrap(self.thumb_size + 8)
				label_font = label.GetFont()
				label_font.SetPointSize(max(7, label_font.GetPointSize() - 1))
				label.SetFont(label_font)

				cell_sizer.Add(bmp_ctrl, 0, wx.ALIGN_CENTER | wx.BOTTOM, 4)
				cell_sizer.Add(label, 0, wx.ALIGN_CENTER)
				cell.SetSizer(cell_sizer)

				cell.Bind(wx.EVT_LEFT_DOWN, lambda event, idx=image_idx: self.on_thumbnail_click(idx))
				bmp_ctrl.Bind(wx.EVT_LEFT_DOWN, lambda event, idx=image_idx: self.on_thumbnail_click(idx))
				label.Bind(wx.EVT_LEFT_DOWN, lambda event, idx=image_idx: self.on_thumbnail_click(idx))

				grid.Add(cell, 0, wx.ALL, 2)
			content_sizer.Add(grid, 0, wx.ALL, 8)

		content_panel.SetSizer(content_sizer)
		content_panel.Layout()
		content_panel.Fit()
		scrolled.SetSizer(wx.BoxSizer(wx.VERTICAL))
		scrolled.GetSizer().Add(content_panel, 0, wx.EXPAND | wx.ALL, 0)
		scrolled.SetVirtualSize(content_panel.GetBestSize())
		scrolled.FitInside()

		cell_width = self.thumb_size + 24
		popup_width = 16 + (cell_width * self.columns)
		content_h = content_panel.GetBestSize().height + 8
		popup_height = min(self.max_height, max(90, content_h))

		scrolled.SetMinSize((popup_width, popup_height))
		root_sizer.Add(scrolled, 1, wx.EXPAND | wx.ALL, 4)
		root_panel.SetSizer(root_sizer)
		root_panel.Layout()
		root_panel.Fit()
		self.SetSize(root_panel.GetBestSize())

	def on_thumbnail_click(self, image_idx):

		self.navigate_callback(image_idx)
		self.Dismiss()


class WindowLv3_BobbysInteractiveEdit(wx.Frame):

	def __init__(self, parent, title, path_to_images, result_path, color_map, aug_methods, model_cp=None, model_cfg=None):

		display_area = wx.Display(0).GetClientArea()
		window_w = max(1000, int(display_area.width * 0.85))
		window_h = max(700, int(display_area.height * 0.85))
		window_w = min(window_w, display_area.width)
		window_h = min(window_h, display_area.height)
		window_x = display_area.x + max(0, int((display_area.width - window_w) / 2))
		window_y = max(30, display_area.y + max(0, int((display_area.height - window_h) / 2)))
		super().__init__(parent, title=title, pos=(window_x, window_y), size=(window_w, window_h))

		self.image_paths = path_to_images
		self.result_path = result_path
		self.color_map = color_map
		self.aug_methods = aug_methods
		self.model_cp = model_cp
		self.model_cfg = model_cfg
		self.current_image_id = 0
		self.current_image = None
		self.current_polygon = []
		self.current_classname = list(self.color_map.keys())[0]
		self.information = read_annotation(os.path.dirname(self.image_paths[0]), color_map=self.color_map)
		self.foreground_points = []
		self.background_points = []
		self.selected_point = None
		self.start_modify = False
		self.show_name = False
		self.AI_help = False
		self.base_scale = 1.0
		self.zoom_scale = 1.0
		self.min_scale = 0.25
		self.max_scale = 8.0
		self.zoom_step = 1.25
		self.draw_offset_x = 0
		self.draw_offset_y = 0
		self.mode = 'REVIEW'
		self.sam2 = None
		self.sam2_loaded = False
		self.status_categories = ['Annotated', 'Un-annotated', 'No Subject', 'Annotator-labeled']
		self.image_status = {}
		self.no_subject_overrides = set()
		self.visited = set()
		self.source_json_images = set()
		self.source_json_annotation_counts = {}
		self.status_buttons = {}
		self.status_more_buttons = {}
		self.status_panel = None
		self.status_default_bg = None
		self.status_default_fg = None
		self.status_default_weight = wx.FONTWEIGHT_NORMAL
		self.thumbnail_popup = None
		self.undo_stack = deque(maxlen=25)
		self.vertex_drag_snapshot = None
		self.vertex_drag_moved = False

		self.base_title = "Bobby's Interactive Edit"
		self.infer_initial_image_statuses()
		self.init_ui()
		self.load_current_image()
		self.update_mode_ui()


	@property
	def scale(self):
		return self.base_scale * self.zoom_scale


	def sam2_model(self):

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.model_cp, device=device))
		return predictor


	def init_ui(self):

		panel = wx.Panel(self)
		vbox = wx.BoxSizer(wx.VERTICAL)
		hbox = wx.BoxSizer(wx.HORIZONTAL)

		self.ai_button = wx.ToggleButton(panel, label='AI Help: OFF', size=(180, 30))
		self.ai_button.Bind(wx.EVT_TOGGLEBUTTON, self.toggle_ai)
		self.ai_placeholder = wx.Panel(panel, size=(180, 30))
		hbox.Add(self.ai_button, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
		hbox.Add(self.ai_placeholder, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
		self.undo_button = wx.Button(panel, label='↩ Undo', size=(110, 30))
		self.undo_button.Bind(wx.EVT_BUTTON, self.on_undo_click)
		hbox.Add(self.undo_button, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)

		self.prev_button = wx.Button(panel, label='← Prev', size=(170, 30))
		self.prev_button.Bind(wx.EVT_BUTTON, self.previous_image)
		hbox.Add(self.prev_button, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)

		self.next_button = wx.Button(panel, label='Next →', size=(170, 30))
		self.next_button.Bind(wx.EVT_BUTTON, self.next_image)
		hbox.Add(self.next_button, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)

		self.delete_button = wx.Button(panel, label='Delete', size=(170, 30))
		self.delete_button.Bind(wx.EVT_BUTTON, self.delete_image)
		hbox.Add(self.delete_button, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)

		self.export_button = wx.Button(panel, label='Export Annotations', size=(170, 30))
		self.export_button.Bind(wx.EVT_BUTTON, self.export_annotations)
		hbox.Add(self.export_button, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)

		self.mode_indicator = wx.StaticText(panel, label='● REVIEW MODE')
		mode_font = self.mode_indicator.GetFont()
		self.mode_indicator.SetFont(wx.Font(mode_font.GetPointSize() + 2, mode_font.GetFamily(), mode_font.GetStyle(), wx.FONTWEIGHT_BOLD))
		hbox.Add(self.mode_indicator, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=6)

		vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.TOP, border=5)
		self.init_status_bar(panel, vbox)

		self.scrolled_canvas = wx.ScrolledWindow(panel, style=wx.VSCROLL | wx.HSCROLL)
		self.scrolled_canvas.SetScrollRate(10, 10)
		self.canvas = wx.Panel(self.scrolled_canvas, pos=(10, 0), size=self.GetSize())
		self.scrolled_canvas.SetBackgroundColour('black')

		self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
		self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_left_click)
		self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.on_right_click)
		self.canvas.Bind(wx.EVT_MOTION, self.on_left_move)
		self.canvas.Bind(wx.EVT_LEFT_UP, self.on_left_up)
		self.canvas.Bind(wx.EVT_MOUSEWHEEL, self.on_mousewheel)
		self.scrolled_canvas.Bind(wx.EVT_SIZE, self.on_canvas_resize)

		self.scrolled_canvas.SetSizer(wx.BoxSizer(wx.VERTICAL))
		self.scrolled_canvas.GetSizer().Add(self.canvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
		vbox.Add(self.scrolled_canvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

		panel.SetSizer(vbox)
		self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)
		self.Show()
		self.update_undo_ui()

	def init_status_bar(self, parent_panel, parent_sizer):

		self.status_panel = wx.Panel(parent_panel, size=(-1, 34))
		status_sizer = wx.BoxSizer(wx.HORIZONTAL)
		for status in self.status_categories:
			group_sizer = wx.BoxSizer(wx.HORIZONTAL)
			button = wx.Button(self.status_panel, label='', size=(165, 30))
			button.Bind(wx.EVT_BUTTON, lambda event, s=status: self.on_status_chip_click(s))
			more_button = wx.Button(self.status_panel, label='...', size=(24, 24))
			more_button.Bind(wx.EVT_BUTTON, lambda event, s=status: self.on_status_more_click(event, s))
			group_sizer.Add(button, 0, wx.ALIGN_CENTER_VERTICAL)
			group_sizer.Add(more_button, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 2)
			status_sizer.Add(group_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
			self.status_buttons[status] = button
			self.status_more_buttons[status] = more_button

		if self.status_buttons:
			first_button = next(iter(self.status_buttons.values()))
			self.status_default_bg = first_button.GetBackgroundColour()
			self.status_default_fg = first_button.GetForegroundColour()
			self.status_default_weight = first_button.GetFont().GetWeight()

		self.status_panel.SetSizer(status_sizer)
		parent_sizer.Add(self.status_panel, flag=wx.ALIGN_CENTER | wx.TOP, border=2)
		self.refresh_status_bar()

	def infer_initial_image_statuses(self):

		self.image_status = {}
		self.no_subject_overrides = set()
		self.source_json_images = set()
		self.source_json_annotation_counts = {}
		if not self.image_paths:
			return

		folder = os.path.dirname(self.image_paths[0])
		annotations_path = os.path.join(folder, 'annotations.json')
		if os.path.exists(annotations_path):
			try:
				with open(annotations_path, 'r') as f:
					annotation_data = json.load(f)
				images_data = annotation_data.get('images', [])
				annotations_data = annotation_data.get('annotations', [])
				image_id_to_name = {}
				for image_record in images_data:
					file_name = image_record.get('file_name')
					image_id = image_record.get('id')
					if file_name is None:
						continue
					self.source_json_images.add(file_name)
					if image_id is not None:
						image_id_to_name[image_id] = file_name

				for annotation_record in annotations_data:
					image_id = annotation_record.get('image_id')
					file_name = image_id_to_name.get(image_id)
					if file_name is None:
						continue
					self.source_json_annotation_counts[file_name] = self.source_json_annotation_counts.get(file_name, 0) + 1
			except Exception:
				self.source_json_images = set()
				self.source_json_annotation_counts = {}

		for image_path in self.image_paths:
			file_name = os.path.basename(image_path)
			if file_name in self.source_json_images:
				if self.source_json_annotation_counts.get(file_name, 0) > 0:
					self.image_status[file_name] = 'Annotated'
				else:
					self.image_status[file_name] = 'No Subject'
			else:
				self.image_status[file_name] = 'Un-annotated'

	def infer_status_from_current_polygons(self, image_name):

		polygons = self.information.get(image_name, {}).get('polygons', [])
		if len(polygons) > 0:
			return 'Annotated'
		return 'Un-annotated'

	def recompute_current_image_status(self):

		if not self.image_paths:
			return
		image_name = os.path.basename(self.image_paths[self.current_image_id])
		if image_name in self.no_subject_overrides:
			self.image_status[image_name] = 'No Subject'
		else:
			self.image_status[image_name] = self.infer_status_from_current_polygons(image_name)
		self.refresh_status_bar()

	def recompute_status_for_image(self, image_name):

		if image_name in self.no_subject_overrides:
			self.image_status[image_name] = 'No Subject'
		else:
			self.image_status[image_name] = self.infer_status_from_current_polygons(image_name)

	def push_undo_snapshot(self, action_type, image_name):

		if image_name not in self.information:
			self.information[image_name] = {'polygons': [], 'class_names': []}
		snapshot = {
			'type': action_type,
			'image_name': image_name,
			'polygons_before': copy.deepcopy(self.information[image_name]['polygons']),
			'class_names_before': copy.deepcopy(self.information[image_name]['class_names']),
			'no_subject_before': image_name in self.no_subject_overrides,
		}
		self.undo_stack.append(snapshot)
		self.update_undo_ui()

	def update_undo_ui(self):

		if hasattr(self, 'undo_button') and self.undo_button is not None:
			self.undo_button.Enable(len(self.undo_stack) > 0)

	def undo_last_action(self):

		if len(self.undo_stack) == 0:
			return
		snapshot = self.undo_stack.pop()
		image_name = snapshot['image_name']
		if image_name not in self.information:
			self.information[image_name] = {'polygons': [], 'class_names': []}
		self.information[image_name]['polygons'] = copy.deepcopy(snapshot['polygons_before'])
		self.information[image_name]['class_names'] = copy.deepcopy(snapshot['class_names_before'])
		if snapshot['no_subject_before']:
			self.no_subject_overrides.add(image_name)
		else:
			self.no_subject_overrides.discard(image_name)
		self.recompute_status_for_image(image_name)
		self.refresh_status_bar()
		self.canvas.Refresh()
		self.update_undo_ui()

	def on_undo_click(self, event):

		self.undo_last_action()
		self.canvas.SetFocus()

	def get_status_counts(self):

		counts = {status: 0 for status in self.status_categories}
		for image_path in self.image_paths:
			file_name = os.path.basename(image_path)
			status = self.image_status.get(file_name, 'Un-annotated')
			if status not in counts:
				status = 'Un-annotated'
			counts[status] += 1
		return counts

	def refresh_status_bar(self):

		if not self.status_buttons:
			return

		counts = self.get_status_counts()
		current_status = None
		if self.image_paths and 0 <= self.current_image_id < len(self.image_paths):
			current_name = os.path.basename(self.image_paths[self.current_image_id])
			current_status = self.image_status.get(current_name, 'Un-annotated')

		for status, button in self.status_buttons.items():
			button.SetLabel('[ ' + status + ': ' + str(counts.get(status, 0)) + ' ]')
			if status == current_status:
				button.SetBackgroundColour(wx.Colour(60, 130, 210))
				button.SetForegroundColour(wx.Colour(255, 255, 255))
				font = button.GetFont()
				font.SetWeight(wx.FONTWEIGHT_BOLD)
				button.SetFont(font)
			else:
				button.SetBackgroundColour(self.status_default_bg)
				button.SetForegroundColour(self.status_default_fg)
				font = button.GetFont()
				font.SetWeight(self.status_default_weight)
				button.SetFont(font)
			button.Refresh()

		self.status_panel.Layout()

	def get_indices_for_status(self, status):

		indices = []
		for idx, image_path in enumerate(self.image_paths):
			file_name = os.path.basename(image_path)
			if self.image_status.get(file_name, 'Un-annotated') == status:
				indices.append(idx)
		return indices

	def navigate_to_image_index(self, target_index):

		if target_index is None:
			return
		if target_index < 0 or target_index >= len(self.image_paths):
			return
		self.current_image_id = target_index
		self.load_current_image()
		self.canvas.SetFocus()
		if self.thumbnail_popup is not None:
			self.thumbnail_popup.Dismiss()
			self.thumbnail_popup = None

	def create_thumbnail_bitmap(self, image_path, max_size=80):

		try:
			pil_image = Image.open(image_path).convert('RGB')
			pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
			width, height = pil_image.size
			wx_image = wx.Image(width, height)
			wx_image.SetData(pil_image.tobytes())
			return wx.Bitmap(wx_image)
		except Exception:
			fallback = wx.Image(max_size, max_size)
			fallback.SetData(bytes([200] * max_size * max_size * 3))
			return wx.Bitmap(fallback)

	def on_status_more_click(self, event, status):

		if self.thumbnail_popup is not None:
			self.thumbnail_popup.Dismiss()
			self.thumbnail_popup = None

		indices = self.get_indices_for_status(status)
		button = self.status_more_buttons.get(status)
		if button is None:
			self.canvas.SetFocus()
			return

		self.thumbnail_popup = ThumbnailGridPopup(self, indices, self.image_paths, self.navigate_to_image_index, thumb_size=80, columns=4, max_height=400)
		self.thumbnail_popup.Position(button.ClientToScreen((0, button.GetSize().height)), (0, 0))
		self.thumbnail_popup.Popup()
		self.canvas.SetFocus()

	def on_status_chip_click(self, status):

		if not self.image_paths:
			return
		indices = self.get_indices_for_status(status)
		if len(indices) == 0:
			self.canvas.SetFocus()
			return

		target_index = None
		for idx in indices:
			if idx not in self.visited:
				target_index = idx
				break
		if target_index is None:
			target_index = indices[0]

		self.navigate_to_image_index(target_index)


	def update_mode_ui(self):

		self.SetTitle(self.base_title + ' - ' + self.mode + ' MODE')
		is_edit = self.mode == 'EDIT'
		self.ai_button.Show(is_edit)
		self.ai_button.Enable(is_edit)
		self.ai_placeholder.Show(not is_edit)
		if not is_edit:
			self.AI_help = False
			self.ai_button.SetValue(False)
			self.ai_button.SetLabel('AI Help: OFF')
			self.mode_indicator.SetLabel('● REVIEW MODE')
			self.mode_indicator.SetForegroundColour(wx.Colour(90, 120, 170))
		else:
			self.mode_indicator.SetLabel('● EDIT MODE')
			self.mode_indicator.SetForegroundColour(wx.Colour(60, 150, 80))
		self.Layout()
		self.canvas.SetFocus()


	def ensure_sam2_loaded(self):

		if self.sam2_loaded:
			return True
		if self.model_cp is None or self.model_cfg is None:
			wx.MessageBox('SAM2 model has not been set up. AI Help remains OFF.', 'AI assistance OFF', wx.ICON_INFORMATION)
			return False

		busy = wx.BusyInfo('Loading AI model, please wait...')
		wx.Yield()
		try:
			self.sam2 = self.sam2_model()
			self.sam2_loaded = True
		finally:
			del busy
		return True


	def set_sam_image_if_needed(self):

		if self.mode != 'EDIT' or not self.AI_help:
			return
		if not self.ensure_sam2_loaded():
			self.AI_help = False
			self.ai_button.SetValue(False)
			self.ai_button.SetLabel('AI Help: OFF')
			return
		image = Image.open(self.image_paths[self.current_image_id])
		image = np.array(image.convert('RGB'))
		self.sam2.set_image(image)


	def toggle_ai(self, event):

		if self.mode != 'EDIT':
			self.ai_button.SetValue(False)
			self.ai_button.SetLabel('AI Help: OFF')
			return

		self.AI_help = self.ai_button.GetValue()
		if self.AI_help:
			if not self.ensure_sam2_loaded():
				self.AI_help = False
				self.ai_button.SetValue(False)
				self.ai_button.SetLabel('AI Help: OFF')
			else:
				self.ai_button.SetLabel('AI Help: ON')
				self.set_sam_image_if_needed()
		else:
			self.ai_button.SetLabel('AI Help: OFF')

		self.canvas.SetFocus()


	def recompute_fit_scale(self):

		if self.current_image is None:
			return
		img_w, img_h = self.current_image.GetSize()
		canvas_w, canvas_h = self.scrolled_canvas.GetClientSize()
		canvas_w = max(canvas_w, 1)
		canvas_h = max(canvas_h, 1)
		self.base_scale = min(canvas_w / max(img_w, 1), canvas_h / max(img_h, 1))


	def update_canvas_geometry(self):

		if self.current_image is None:
			return
		self.recompute_fit_scale()
		scaled_w = max(1, int(self.current_image.GetWidth() * self.scale))
		scaled_h = max(1, int(self.current_image.GetHeight() * self.scale))
		self.scrolled_canvas.SetVirtualSize((scaled_w, scaled_h))
		self.canvas.SetSize((scaled_w, scaled_h))


	def load_current_image(self):

		if self.image_paths:
			path = self.image_paths[self.current_image_id]
			self.current_image = wx.Image(path, wx.BITMAP_TYPE_ANY)
			image_name = os.path.basename(path)
			if image_name not in self.information:
				self.information[image_name] = {'polygons': [], 'class_names': []}
			self.current_polygon = []
			self.foreground_points = []
			self.background_points = []
			self.zoom_scale = 1.0
			self.update_canvas_geometry()
			self.scrolled_canvas.Scroll(0, 0)
			self.canvas.Refresh()
			self.set_sam_image_if_needed()
			self.visited.add(self.current_image_id)
			image_name = os.path.basename(path)
			if image_name in self.no_subject_overrides:
				self.image_status[image_name] = 'No Subject'
			elif image_name not in self.image_status:
				self.image_status[image_name] = self.infer_status_from_current_polygons(image_name)
			self.refresh_status_bar()


	def previous_image(self, event):

		if self.image_paths and self.current_image_id > 0:
			self.current_image_id -= 1
			self.load_current_image()
		self.canvas.SetFocus()


	def next_image(self, event):

		if self.image_paths and self.current_image_id < len(self.image_paths) - 1:
			self.current_image_id += 1
			self.load_current_image()
		self.canvas.SetFocus()


	def delete_image(self, event):

		if self.image_paths:
			removed_index = self.current_image_id
			path = self.image_paths[self.current_image_id]
			self.image_paths.remove(path)
			image_name = os.path.basename(path)
			if image_name in self.information:
				del self.information[image_name]
			if image_name in self.image_status:
				del self.image_status[image_name]
			if image_name in self.no_subject_overrides:
				self.no_subject_overrides.remove(image_name)
			new_visited = set()
			for idx in self.visited:
				if idx < removed_index:
					new_visited.add(idx)
				elif idx > removed_index:
					new_visited.add(idx - 1)
			self.visited = new_visited
			if len(self.image_paths) == 0:
				self.current_image = None
				self.canvas.Refresh()
				self.refresh_status_bar()
				return
			if self.current_image_id >= len(self.image_paths):
				self.current_image_id = len(self.image_paths) - 1
			self.load_current_image()
		self.canvas.SetFocus()


	def image_coords_from_event(self, event):

		x, y = event.GetX(), event.GetY()
		scale = self.scale
		if scale <= 0:
			return None
		xi = int(x / scale)
		yi = int(y / scale)
		img_w, img_h = self.current_image.GetSize()
		if xi < 0 or yi < 0 or xi >= img_w or yi >= img_h:
			return None
		return xi, yi


	def on_paint(self, event):

		if self.current_image is None:
			return

		dc = wx.PaintDC(self.canvas)
		w, h = self.current_image.GetSize()
		scaled_image = self.current_image.Scale(max(1, int(w * self.scale)), max(1, int(h * self.scale)), wx.IMAGE_QUALITY_HIGH)
		dc.DrawBitmap(wx.Bitmap(scaled_image), 0, 0, True)
		image_name = os.path.basename(self.image_paths[self.current_image_id])
		polygons = self.information[image_name]['polygons']
		class_names = self.information[image_name]['class_names']

		if len(polygons) > 0:
			for i, polygon in enumerate(polygons):
				color = self.color_map[class_names[i]]
				pen = wx.Pen(wx.Colour(*color), width=2)
				dc.SetPen(pen)
				dc.DrawLines([(int(x * self.scale), int(y * self.scale)) for x, y in polygon])
				if self.start_modify and self.mode == 'EDIT':
					brush = wx.Brush(wx.Colour(*color))
					dc.SetBrush(brush)
					for x, y in polygon:
						dc.DrawCircle(int(x * self.scale), int(y * self.scale), 4)
				if self.show_name:
					x_max = int(max(x for x, y in polygon) * self.scale)
					x_min = int(min(x for x, y in polygon) * self.scale)
					y_max = int(max(y for x, y in polygon) * self.scale)
					y_min = int(min(y for x, y in polygon) * self.scale)
					cx = int((x_max + x_min) / 2)
					cy = int((y_max + y_min) / 2)
					dc.SetTextForeground(wx.Colour(*color))
					dc.SetFont(wx.Font(wx.FontInfo(15).FaceName('Arial')))
					dc.DrawText(str(class_names[i]), cx, cy)

		if len(self.current_polygon) > 0:
			current_polygon = [i for i in self.current_polygon]
			current_polygon.append(current_polygon[0])
			color = self.color_map[self.current_classname]
			brush = wx.Brush(wx.Colour(*color))
			dc.SetBrush(brush)
			for x, y in current_polygon:
				dc.DrawCircle(int(x * self.scale), int(y * self.scale), 4)
			pen = wx.Pen(wx.Colour(*color), width=2)
			dc.SetPen(pen)
			dc.DrawLines([(int(x * self.scale), int(y * self.scale)) for x, y in current_polygon])


	def on_left_click(self, event):

		if self.mode != 'EDIT':
			return
		if self.current_image is None:
			return
		coords = self.image_coords_from_event(event)
		if coords is None:
			return
		x, y = coords

		if self.start_modify:

			image_name = os.path.basename(self.image_paths[self.current_image_id])
			for i, polygon in enumerate(self.information[image_name]['polygons']):
				for j, (px, py) in enumerate(polygon):
					if abs(px - x) < 5 / self.scale and abs(py - y) < 5 / self.scale:
						self.selected_point = (polygon, j, i)
						self.vertex_drag_snapshot = {
							'type': 'vertex_drag',
							'image_name': image_name,
							'polygons_before': copy.deepcopy(self.information[image_name]['polygons']),
							'class_names_before': copy.deepcopy(self.information[image_name]['class_names']),
							'no_subject_before': image_name in self.no_subject_overrides,
						}
						self.vertex_drag_moved = False
						return

		else:

			if self.AI_help:
				self.foreground_points.append([x, y])
				points = self.foreground_points + self.background_points
				labels = [1 for i in range(len(self.foreground_points))] + [0 for i in range(len(self.background_points))]
				masks, scores, logits = self.sam2.predict(point_coords=np.array(points), point_labels=np.array(labels))
				mask = masks[np.argsort(scores)[::-1]][0]
				self.current_polygon = mask_to_polygon(mask)
			else:
				self.current_polygon.append((x, y))

		self.canvas.Refresh()


	def on_right_click(self, event):

		if self.mode != 'EDIT':
			return
		if self.current_image is None:
			return
		coords = self.image_coords_from_event(event)
		if coords is None:
			return
		x, y = coords

		if self.start_modify:

			return

		else:

			if len(self.current_polygon) > 0:

				if self.AI_help:
					self.background_points.append([x, y])
					points = self.foreground_points + self.background_points
					labels = [1 for i in range(len(self.foreground_points))] + [0 for i in range(len(self.background_points))]
					masks, scores, logits = self.sam2.predict(point_coords=np.array(points), point_labels=np.array(labels))
					mask = masks[np.argsort(scores)[::-1]][0]
					self.current_polygon = mask_to_polygon(mask)
				else:
					self.current_polygon.pop()

			else:

				to_delete = []
				image_name = os.path.basename(self.image_paths[self.current_image_id])
				polygons = self.information[image_name]['polygons']
				if len(polygons) > 0:
					for i, polygon in enumerate(polygons):
						x_max = max(px for px, py in polygon)
						x_min = min(px for px, py in polygon)
						y_max = max(py for px, py in polygon)
						y_min = min(py for px, py in polygon)
						if x_min <= x <= x_max and y_min <= y <= y_max:
							to_delete.append(i)
				if len(to_delete) > 0:
					self.push_undo_snapshot('delete', image_name)
					for i in sorted(to_delete, reverse=True):
						del self.information[image_name]['polygons'][i]
						del self.information[image_name]['class_names'][i]
					self.recompute_current_image_status()

		self.canvas.Refresh()


	def on_key_press(self, event):

		key_code = event.GetKeyCode()
		if key_code in (ord('Z'), ord('z')) and (event.CmdDown() or event.ControlDown()):
			self.undo_last_action()
			return

		if key_code == wx.WXK_TAB:
			if self.mode == 'REVIEW':
				self.mode = 'EDIT'
				self.AI_help = True
				self.ai_button.SetValue(True)
				self.ai_button.SetLabel('AI Help: ON')
				if not self.ensure_sam2_loaded():
					self.AI_help = False
					self.ai_button.SetValue(False)
					self.ai_button.SetLabel('AI Help: OFF')
				self.set_sam_image_if_needed()
			else:
				self.mode = 'REVIEW'
				self.current_polygon = []
				self.foreground_points = []
				self.background_points = []
				self.start_modify = False
			self.update_mode_ui()
			self.canvas.Refresh()
			return

		if key_code == wx.WXK_LEFT:
			self.previous_image(None)
			return
		if key_code == wx.WXK_RIGHT:
			self.next_image(None)
			return
		if key_code == wx.WXK_SPACE:
			self.show_name = not self.show_name
			self.canvas.Refresh()
			return
		if key_code in (ord('N'), ord('n')):
			if self.image_paths:
				image_name = os.path.basename(self.image_paths[self.current_image_id])
				self.push_undo_snapshot('no_subject', image_name)
				if image_name in self.no_subject_overrides:
					self.no_subject_overrides.remove(image_name)
					self.image_status[image_name] = self.infer_status_from_current_polygons(image_name)
				else:
					self.no_subject_overrides.add(image_name)
					self.image_status[image_name] = 'No Subject'
				self.refresh_status_bar()
			return

		if self.mode != 'EDIT':
			event.Skip()
			return

		if key_code == wx.WXK_RETURN:
			if len(self.current_polygon) > 2:
				image_name = os.path.basename(self.image_paths[self.current_image_id])
				classnames = sorted(list(self.color_map.keys()))
				current_index = classnames.index(self.current_classname)
				dialog = wx.SingleChoiceDialog(self, message='Choose object class name', caption='Class Name', choices=classnames)
				dialog.SetSelection(current_index)
				if dialog.ShowModal() == wx.ID_OK:
					self.current_classname = dialog.GetStringSelection()
					if len(self.current_polygon) > 0:
						self.push_undo_snapshot('commit', image_name)
						self.current_polygon.append(self.current_polygon[0])
						self.information[image_name]['polygons'].append(self.current_polygon)
						self.information[image_name]['class_names'].append(self.current_classname)
				dialog.Destroy()
				self.current_polygon = []
				self.foreground_points = []
				self.background_points = []
				self.recompute_current_image_status()
				self.canvas.Refresh()
		elif key_code == wx.WXK_SHIFT:
			self.start_modify = not self.start_modify
			self.canvas.Refresh()
		elif key_code == wx.WXK_ESCAPE:
			self.current_polygon = []
			self.foreground_points = []
			self.background_points = []
			self.canvas.Refresh()
		else:
			event.Skip()


	def on_left_move(self, event):

		if self.mode != 'EDIT':
			return
		if self.selected_point is not None and event.Dragging() and event.LeftIsDown():
			polygon, j, i = self.selected_point
			coords = self.image_coords_from_event(event)
			if coords is None:
				return
			x, y = coords
			if polygon[j] != (x, y):
				self.vertex_drag_moved = True
			polygon[j] = (x, y)
			image_name = os.path.basename(self.image_paths[self.current_image_id])
			self.information[image_name]['polygons'][i] = polygon


	def on_left_up(self, event):

		if self.selected_point is not None and self.vertex_drag_moved and self.vertex_drag_snapshot is not None:
			self.undo_stack.append(self.vertex_drag_snapshot)
			self.update_undo_ui()
		self.vertex_drag_snapshot = None
		self.vertex_drag_moved = False
		self.selected_point = None
		self.canvas.Refresh()


	def on_mousewheel(self, event):

		if self.current_image is None:
			return

		if self.start_modify:
			return

		rotation = event.GetWheelRotation()
		if rotation > 0:
			self.zoom_scale = min(self.zoom_scale * self.zoom_step, self.max_scale)
		else:
			self.zoom_scale = max(self.zoom_scale / self.zoom_step, self.min_scale)

		self.update_canvas_geometry()
		self.canvas.Refresh()


	def on_canvas_resize(self, event):

		self.update_canvas_geometry()
		self.canvas.Refresh()
		event.Skip()


	def export_annotations(self, event):

		if not self.information:
			wx.MessageBox('No annotations to export.', 'Error', wx.ICON_ERROR)
			return

		generate_annotation(os.path.dirname(self.image_paths[0]), self.information, self.result_path, self.result_path, self.aug_methods, self.color_map)
		generate_annotation(os.path.dirname(self.image_paths[0]), self.information, os.path.dirname(self.image_paths[0]), self.result_path, [], self.color_map)

		wx.MessageBox('Annotations exported successfully.', 'Success', wx.ICON_INFORMATION)

		self.canvas.SetFocus()
