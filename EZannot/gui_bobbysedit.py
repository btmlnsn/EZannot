import os
import cv2
import wx
import json
import random
import torch
import copy
import gc
import numpy as np
from collections import deque
from pathlib import Path
from PIL import Image
from screeninfo import get_monitors
from EZannot.sam2.build_sam import build_sam2
from EZannot.sam2.sam2_image_predictor import SAM2ImagePredictor
from .annotator import Annotator
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
		self.annotator_model_path = None
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

		module_annotator_model = wx.BoxSizer(wx.HORIZONTAL)
		button_annotator_model = wx.Button(panel, label='Select a trained Annotator\nfor Auto-Annotate', size=(300, 40))
		button_annotator_model.Bind(wx.EVT_BUTTON, self.select_annotator_model)
		wx.Button.SetToolTip(button_annotator_model, 'Select a trained Annotator model folder for Auto-Annotate.')
		self.text_annotator_model = wx.StaticText(panel, label='None.', style=wx.ALIGN_LEFT | wx.ST_ELLIPSIZE_END)
		module_annotator_model.Add(button_annotator_model, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		module_annotator_model.Add(self.text_annotator_model, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
		boxsizer.Add(module_annotator_model, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
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
			self.text_input.SetLabel(selected_folder + ' - ' + str(len(self.path_to_images)) + ' images found')
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

	def select_annotator_model(self, event):

		dialog = wx.DirDialog(self, 'Select a trained Annotator model folder', '', style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal() == wx.ID_OK:
			self.annotator_model_path = dialog.GetPath()
			self.text_annotator_model.SetLabel('Annotator model: ' + self.annotator_model_path)
		dialog.Destroy()


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
				annotator_model_path=self.annotator_model_path,
			)


class FilenameListPopup(wx.PopupTransientWindow):

	def __init__(self, parent_frame, indices, image_paths, navigate_callback, popup_width=300, max_height=400):

		super().__init__(parent_frame)
		self.parent_frame = parent_frame
		self.indices = indices
		self.image_paths = image_paths
		self.navigate_callback = navigate_callback

		root_panel = wx.Panel(self)
		root_sizer = wx.BoxSizer(wx.VERTICAL)
		self.listbox = wx.ListBox(root_panel, style=wx.LB_SINGLE)
		for image_idx in self.indices:
			self.listbox.Append(os.path.basename(self.image_paths[image_idx]))
		self.listbox.Bind(wx.EVT_LISTBOX, self.on_select)

		item_height = max(18, self.listbox.GetCharHeight() + 8)
		content_height = max(40, item_height * max(1, len(self.indices)))
		list_height = min(max_height, content_height)
		self.listbox.SetMinSize((popup_width, list_height))
		root_sizer.Add(self.listbox, 1, wx.EXPAND | wx.ALL, 4)
		root_panel.SetSizer(root_sizer)
		root_panel.Layout()
		root_panel.Fit()
		self.SetSize((popup_width + 8, list_height + 8))

	def on_select(self, event):

		selected = event.GetSelection()
		if selected != wx.NOT_FOUND and selected < len(self.indices):
			self.navigate_callback(self.indices[selected])
		self.Dismiss()


class WindowLv3_BobbysInteractiveEdit(wx.Frame):

	def __init__(self, parent, title, path_to_images, result_path, color_map, aug_methods, model_cp=None, model_cfg=None, annotator_model_path=None):

		display_area = wx.Display(0).GetClientArea()
		window_w = max(1000, int(display_area.width * 0.85))
		window_h = max(700, int(display_area.height * 0.85))
		window_w = min(window_w, display_area.width)
		window_h = min(window_h, display_area.height)
		window_x = display_area.x + max(0, int((display_area.width - window_w) / 2))
		window_y = max(30, display_area.y + max(0, int((display_area.height - window_h) / 2)))
		style = wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
		super().__init__(parent, title=title, pos=(window_x, window_y), size=(window_w, window_h), style=style)

		self.image_paths = path_to_images
		self.result_path = result_path
		self.color_map = color_map
		self.aug_methods = aug_methods
		self.model_cp = model_cp
		self.model_cfg = model_cfg
		self.annotator_model_path = annotator_model_path
		if self.model_cp:
			self.sam_model_name = os.path.splitext(os.path.basename(self.model_cp))[0]
		else:
			self.sam_model_name = 'SAM model'
		self.sam2_last_error = None
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
		self.sam2_loading_started = False
		self.ai_model_mode = ''
		self.annotator_model = None
		self.status_categories = ['Annotated', 'Un-annotated', 'No Subject', 'Annotator-labeled']
		self.image_status = {}
		self.active_category = None
		self.annotations_complete_mode = False
		self.flash_message_image_name = None
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
		toolbar_sizer = wx.BoxSizer(wx.VERTICAL)
		top_row = wx.BoxSizer(wx.HORIZONTAL)
		bottom_row = wx.BoxSizer(wx.HORIZONTAL)

		self.filename_label = wx.StaticText(panel, label='', style=wx.ALIGN_LEFT)
		self.filename_label.SetMinSize((220, -1))
		self.filename_label.SetMaxSize((220, -1))

		self.prev_button = wx.Button(panel, label='← Prev', size=(170, 30))
		self.prev_button.Bind(wx.EVT_BUTTON, self.previous_image)

		self.next_button = wx.Button(panel, label='Next →', size=(170, 30))
		self.next_button.Bind(wx.EVT_BUTTON, self.next_image)

		self.delete_button = wx.Button(panel, label='Delete', size=(170, 30))
		self.delete_button.Bind(wx.EVT_BUTTON, self.delete_image)

		self.export_button = wx.Button(panel, label='Export Annotations', size=(170, 30))
		self.export_button.Bind(wx.EVT_BUTTON, self.export_annotations)

		self.ai_model_label = wx.StaticText(panel, label=f'{self.sam_model_name} loaded:')
		self.ai_model_choice = wx.Choice(panel, choices=['Off', '', 'On'])
		self.ai_model_choice.SetSelection(1)
		self.ai_model_choice.Bind(wx.EVT_CHOICE, self.on_ai_model_choice_changed)

		self.undo_button = wx.Button(panel, label='Undo', size=(110, 30))
		self.undo_button.Bind(wx.EVT_BUTTON, self.on_undo_click)

		self.auto_annotate_button = wx.Button(panel, label='Auto-Annotate', size=(170, 30))
		self.auto_annotate_button.Bind(wx.EVT_BUTTON, self.run_auto_annotate)
		self.auto_annotate_button.Enable(self.annotator_model_path is not None)

		self.mode_indicator = wx.StaticText(panel, label='● Review Mode')
		mode_font = self.mode_indicator.GetFont()
		self.mode_indicator.SetFont(wx.Font(mode_font.GetPointSize() + 2, mode_font.GetFamily(), mode_font.GetStyle(), wx.FONTWEIGHT_BOLD))
		review_size = self.mode_indicator.GetBestSize()
		self.mode_indicator.SetLabel('● Edit Mode')
		edit_size = self.mode_indicator.GetBestSize()
		self.mode_indicator.SetLabel('● Review Mode')
		mode_min_w = max(review_size.width, edit_size.width) + 6
		self.mode_indicator.SetMinSize((mode_min_w, -1))

		center_top = wx.BoxSizer(wx.HORIZONTAL)
		center_top.AddStretchSpacer(1)
		center_top.Add(self.prev_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
		center_top.Add(self.next_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
		center_top.Add(self.delete_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
		center_top.Add(self.export_button, 0, wx.ALIGN_CENTER_VERTICAL)
		center_top.AddStretchSpacer(1)

		ai_row = wx.BoxSizer(wx.HORIZONTAL)
		ai_row.AddStretchSpacer(1)
		ai_row.Add(self.ai_model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
		ai_row.Add(self.ai_model_choice, 0, wx.ALIGN_CENTER_VERTICAL)

		mode_row = wx.BoxSizer(wx.HORIZONTAL)
		mode_row.AddStretchSpacer(1)
		mode_row.Add(self.mode_indicator, 0, wx.ALIGN_CENTER_VERTICAL)

		top_right_stack = wx.BoxSizer(wx.VERTICAL)
		top_right_stack.Add(ai_row, 0, wx.EXPAND)
		top_right_stack.AddSpacer(8)
		top_right_stack.Add(mode_row, 0, wx.EXPAND)

		top_right_width = max(mode_min_w, self.ai_model_label.GetBestSize().width + self.ai_model_choice.GetBestSize().width + 12) + 12

		top_row.Add(self.filename_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
		top_row.Add(center_top, 1, wx.EXPAND)
		top_row.Add(top_right_stack, 0, wx.EXPAND | wx.RIGHT, 12)

		center_buttons = wx.BoxSizer(wx.HORIZONTAL)
		center_buttons.AddStretchSpacer(1)
		center_buttons.Add(self.undo_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
		center_buttons.Add(self.auto_annotate_button, 0, wx.ALIGN_CENTER_VERTICAL)
		center_buttons.AddStretchSpacer(1)

		bottom_left_spacer = wx.Panel(panel)
		bottom_left_spacer.SetMinSize((220, -1))

		bottom_right_spacer = wx.Panel(panel)
		bottom_right_spacer.SetMinSize((top_right_width, -1))

		bottom_row.Add(bottom_left_spacer, 0, wx.EXPAND)
		bottom_row.Add(center_buttons, 1, wx.EXPAND)
		bottom_row.Add(bottom_right_spacer, 0, wx.EXPAND)

		toolbar_sizer.Add(top_row, 0, wx.EXPAND | wx.BOTTOM, 4)
		toolbar_sizer.Add(bottom_row, 0, wx.EXPAND)
		vbox.Add(toolbar_sizer, flag=wx.EXPAND | wx.TOP, border=5)

		annotation_presence_row = wx.BoxSizer(wx.HORIZONTAL)
		self.annotation_presence_label = wx.StaticText(panel, label='', style=wx.ALIGN_LEFT)
		indicator_font = self.annotation_presence_label.GetFont()
		self.annotation_presence_label.SetFont(wx.Font(int(indicator_font.GetPointSize()) + 1, indicator_font.GetFamily(), indicator_font.GetStyle(), wx.FONTWEIGHT_BOLD))
		self.annotation_presence_label.SetMinSize((-1, 22))
		self.annotation_presence_label.SetMaxSize((-1, 22))
		self.annotation_presence_label.Hide()
		annotation_presence_row.Add(self.annotation_presence_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
		vbox.Add(annotation_presence_row, 0, wx.EXPAND)

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

		self.flash_panel = wx.Panel(self.canvas, style=wx.BORDER_NONE)
		self.flash_panel.SetBackgroundColour(wx.Colour(45, 45, 48))
		self.flash_text = wx.StaticText(self.flash_panel, label='')
		self.flash_text.SetForegroundColour(wx.Colour(255, 255, 255))
		flash_font = self.flash_text.GetFont()
		self.flash_text.SetFont(wx.Font(int(flash_font.GetPointSize()) + 1, flash_font.GetFamily(), flash_font.GetStyle(), flash_font.GetWeight()))
		flash_sizer = wx.BoxSizer(wx.VERTICAL)
		flash_sizer.Add(self.flash_text, 0, wx.ALL, 8)
		self.flash_panel.SetSizer(flash_sizer)
		self.flash_panel.Hide()
		self.canvas.Bind(wx.EVT_SIZE, self.on_canvas_child_layout)

		panel.SetSizer(vbox)
		self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)
		self.Show()
		self.update_undo_ui()
		self.apply_ai_model_mode()

	def on_canvas_child_layout(self, event):

		self._position_flash_panel()
		event.Skip()

	def _position_flash_panel(self):

		if not hasattr(self, 'flash_panel'):
			return
		self.flash_panel.Fit()
		cw, ch = self.canvas.GetClientSize()
		fw, fh = self.flash_panel.GetSize()
		self.flash_panel.SetPosition((12, max(0, ch - fh - 12)))

	def update_annotation_presence_indicator(self):

		if not hasattr(self, 'annotation_presence_label'):
			return
		image_name = self.get_current_image_name()
		if image_name is None:
			self.annotation_presence_label.SetLabel('')
			self.annotation_presence_label.Hide()
			self._layout_annotation_presence_parent()
			return
		status = self.image_status.get(image_name, 'Un-annotated')
		if status != 'Annotator-labeled':
			self.annotation_presence_label.SetLabel('')
			self.annotation_presence_label.Hide()
			self._layout_annotation_presence_parent()
			return
		polygons = self.information.get(image_name, {}).get('polygons', [])
		if len(polygons) > 0:
			self.annotation_presence_label.SetLabel('Annotation present')
			self.annotation_presence_label.SetForegroundColour(wx.Colour(70, 185, 95))
		else:
			self.annotation_presence_label.SetLabel('No annotation present')
			self.annotation_presence_label.SetForegroundColour(wx.Colour(230, 170, 55))
		self.annotation_presence_label.Show()
		self._layout_annotation_presence_parent()

	def _layout_annotation_presence_parent(self):

		parent = self.annotation_presence_label.GetParent()
		if parent is not None:
			parent.Layout()
		self.Layout()

	def show_flash_message(self, message):

		self.flash_text.SetLabel(message)
		self.flash_message_image_name = self.get_current_image_name()
		self.flash_panel.Fit()
		self._position_flash_panel()
		self.flash_panel.Show()
		self.flash_panel.Raise()

	def hide_flash_message(self):

		self.flash_panel.Hide()
		self.flash_message_image_name = None
		self.canvas.Refresh()

	def dismiss_annotations_complete(self):

		if self.annotations_complete_mode:
			self.annotations_complete_mode = False
			self.canvas.Refresh()
			self._position_flash_panel()

	def on_ai_model_choice_changed(self, event):

		print(f"SAM switch changed to: {event.GetString()!r}")
		self.apply_ai_model_mode()
		self.canvas.SetFocus()

	def apply_ai_model_mode(self):

		previous_mode = self.ai_model_mode
		selection = self.ai_model_choice.GetStringSelection() if hasattr(self, 'ai_model_choice') else ''
		self.ai_model_mode = selection
		mode_changed = previous_mode != self.ai_model_mode
		if self.ai_model_mode == 'Off':
			if self.sam2 is not None:
				del self.sam2
				if 'sam2_model' in self.__dict__:
					del self.sam2_model
				torch.cuda.empty_cache()
				gc.collect()
				self.sam2 = None
				self.sam2_loaded = False
				self.AI_help = False
				print(f'{self.sam_model_name} unloaded from memory.')
			else:
				self.AI_help = False
				if mode_changed:
					print(f'{self.sam_model_name} already unloaded, nothing to do.')
			return
		if self.ai_model_mode == '':
			if mode_changed:
				self.sam2_loaded = False
				print(f'AI Model set to Auto. {self.sam_model_name} will load on next Edit entry.')
			return
		if self.ai_model_mode == 'On' and not self.sam2_loaded:
			print(f'AI Model set to On. Loading {self.sam_model_name} now...')
			if self.ensure_sam2_loaded():
				print(f'{self.sam_model_name} loaded into memory.')
			else:
				err = self.sam2_last_error
				print(f'{self.sam_model_name} failed to load: {err}')

	def set_filename_label(self, filename):

		if filename is None:
			filename = ''
		basename = os.path.basename(filename) if filename else ''
		if basename == '':
			self.filename_label.SetLabel('')
			return
		if len(basename) > 28:
			display = basename[:12] + '...' + basename[-10:]
		else:
			display = basename
		self.filename_label.SetLabel(display)

	def status_chip_display_name(self, status):

		if status == 'Annotator-labeled':
			return 'AI-labeled'
		return status

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
			'status_before': self.image_status.get(image_name),
		}
		self.undo_stack.append(snapshot)
		self.update_undo_ui()

	def push_accept_undo_snapshot(self, image_name):

		if image_name not in self.information:
			self.information[image_name] = {'polygons': [], 'class_names': []}
		snapshot = {
			'type': 'accept',
			'image_name': image_name,
			'status_before': 'Annotator-labeled',
			'polygons_before': copy.deepcopy(self.information[image_name]['polygons']),
			'class_names_before': copy.deepcopy(self.information[image_name]['class_names']),
			'no_subject_before': image_name in self.no_subject_overrides,
		}
		self.undo_stack.append(snapshot)
		self.update_undo_ui()

	def _undo_action_type_label(self, action_type):

		labels = {
			'commit': 'Commit',
			'delete': 'Delete',
			'no_subject': 'No Subject',
			'vertex_drag': 'Vertex edit',
			'annotator_batch': 'Auto-Annotate',
		}
		return labels.get(action_type, str(action_type))

	def update_undo_ui(self):

		if hasattr(self, 'undo_button') and self.undo_button is not None:
			self.undo_button.Enable(len(self.undo_stack) > 0)

	def undo_last_action(self):

		if len(self.undo_stack) == 0:
			return
		snapshot = self.undo_stack.pop()
		if snapshot.get('type') == 'annotator_batch':
			for image_name, payload in snapshot.get('images_before', {}).items():
				if image_name not in self.information:
					self.information[image_name] = {'polygons': [], 'class_names': []}
				self.information[image_name]['polygons'] = copy.deepcopy(payload.get('polygons_before', []))
				self.information[image_name]['class_names'] = copy.deepcopy(payload.get('class_names_before', []))
				if payload.get('no_subject_before', False):
					self.no_subject_overrides.add(image_name)
				else:
					self.no_subject_overrides.discard(image_name)
				self.image_status[image_name] = payload.get('status_before', 'Un-annotated')
			self.refresh_status_bar()
			self.canvas.Refresh()
			self.update_undo_ui()
			self.show_flash_message('Undone: ' + self._undo_action_type_label('annotator_batch') + '. Review the image.')
			return

		if snapshot.get('type') == 'accept':
			image_name = snapshot['image_name']
			if image_name not in self.information:
				self.information[image_name] = {'polygons': [], 'class_names': []}
			self.information[image_name]['polygons'] = copy.deepcopy(snapshot['polygons_before'])
			self.information[image_name]['class_names'] = copy.deepcopy(snapshot['class_names_before'])
			if snapshot['no_subject_before']:
				self.no_subject_overrides.add(image_name)
			else:
				self.no_subject_overrides.discard(image_name)
			self.image_status[image_name] = 'Annotator-labeled'
			self.refresh_status_bar()
			self.canvas.Refresh()
			self.update_undo_ui()
			self.show_flash_message('Undone: Accept. Image returned to AI-labeled.')
			return

		image_name = snapshot['image_name']
		if image_name not in self.information:
			self.information[image_name] = {'polygons': [], 'class_names': []}
		self.information[image_name]['polygons'] = copy.deepcopy(snapshot['polygons_before'])
		self.information[image_name]['class_names'] = copy.deepcopy(snapshot['class_names_before'])
		if snapshot['no_subject_before']:
			self.no_subject_overrides.add(image_name)
		else:
			self.no_subject_overrides.discard(image_name)
		if snapshot.get('status_before') == 'Annotator-labeled':
			self.image_status[image_name] = 'Annotator-labeled'
		else:
			self.recompute_status_for_image(image_name)
		self.refresh_status_bar()
		self.canvas.Refresh()
		self.update_undo_ui()
		self.show_flash_message('Undone: ' + self._undo_action_type_label(snapshot['type']) + '. Review the image.')

	def on_undo_click(self, event):

		self.undo_last_action()
		self.hide_flash_message()
		self.canvas.SetFocus()

	def get_current_image_name(self):

		if not self.image_paths:
			return None
		return os.path.basename(self.image_paths[self.current_image_id])

	def finalize_status_after_edit(self, image_name):

		if image_name is None:
			return
		if image_name in self.no_subject_overrides:
			self.image_status[image_name] = 'No Subject'
			return
		if image_name not in self.information:
			self.information[image_name] = {'polygons': [], 'class_names': []}
		polygons = self.information[image_name]['polygons']
		current_status = self.image_status.get(image_name, 'Un-annotated')
		if current_status == 'Annotator-labeled':
			if len(polygons) == 0:
				self.image_status[image_name] = 'Un-annotated'
			else:
				self.image_status[image_name] = 'Annotated'
		else:
			self.image_status[image_name] = self.infer_status_from_current_polygons(image_name)

	def run_auto_annotate(self, event):

		if not self.image_paths:
			wx.MessageBox('No images selected for annotation.', 'Error', wx.OK | wx.ICON_ERROR)
			return

		if self.annotator_model_path is None:
			wx.MessageBox('No Annotator model selected in setup.', 'Error', wx.OK | wx.ICON_ERROR)
			self.canvas.SetFocus()
			return
		path_to_annotator = self.annotator_model_path

		model_parameters_path = os.path.join(path_to_annotator, 'model_parameters.txt')
		if not os.path.exists(model_parameters_path):
			wx.MessageBox('Invalid Annotator folder: model_parameters.txt is missing.', 'Error', wx.OK | wx.ICON_ERROR)
			self.canvas.SetFocus()
			return

		try:
			with open(model_parameters_path, 'r') as f:
				model_parameters = json.loads(f.read())
			object_kinds = model_parameters.get('object_names', [])
			self.annotator_model = Annotator()
			self.annotator_model.load(path_to_annotator, object_kinds)
		except Exception as exc:
			wx.MessageBox('Failed to load Annotator model:\n' + str(exc), 'Error', wx.OK | wx.ICON_ERROR)
			self.canvas.SetFocus()
			return

		targets = []
		for path in self.image_paths:
			image_name = os.path.basename(path)
			if image_name not in self.information:
				self.information[image_name] = {'polygons': [], 'class_names': []}
			if self.image_status.get(image_name, 'Un-annotated') == 'Un-annotated':
				targets.append((path, image_name))

		if len(targets) == 0:
			wx.MessageBox('No Un-annotated images are eligible for Auto-Annotate.', 'Info', wx.OK | wx.ICON_INFORMATION)
			self.canvas.SetFocus()
			return

		batch_snapshot = {'type': 'annotator_batch', 'images_before': {}}
		for _, image_name in targets:
			batch_snapshot['images_before'][image_name] = {
				'polygons_before': copy.deepcopy(self.information[image_name]['polygons']),
				'class_names_before': copy.deepcopy(self.information[image_name]['class_names']),
				'no_subject_before': image_name in self.no_subject_overrides,
				'status_before': self.image_status.get(image_name, 'Un-annotated'),
			}
		self.undo_stack.append(batch_snapshot)
		self.update_undo_ui()

		progress = wx.ProgressDialog(
			'Auto-Annotate',
			'Running annotator...',
			maximum=len(targets),
			parent=self,
			style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_AUTO_HIDE
		)

		try:
			for idx, (image_path, image_name) in enumerate(targets, start=1):
				image = cv2.imread(image_path)
				if image is None:
					image = np.array(Image.open(image_path).convert('RGB'))
					image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				output = self.annotator_model.inference([{'image': torch.as_tensor(image.astype('float32').transpose(2, 0, 1))}])
				instances = output[0]['instances'].to('cpu')
				masks = instances.pred_masks.numpy().astype(np.uint8)
				classes = instances.pred_classes.numpy()
				classes = [self.annotator_model.object_mapping[str(c)] for c in classes]

				polygons = []
				class_names = []
				for mask, class_name in zip(masks, classes):
					polygon = mask_to_polygon(mask)
					if len(polygon) > 2:
						polygons.append(polygon)
						class_names.append(class_name)

				self.information[image_name]['polygons'] = polygons
				self.information[image_name]['class_names'] = class_names
				self.image_status[image_name] = 'Annotator-labeled'

				progress.Update(idx, 'Running annotator... (' + str(idx) + '/' + str(len(targets)) + ')')
				wx.YieldIfNeeded()
		finally:
			progress.Destroy()

		self.refresh_status_bar()
		self.canvas.Refresh()
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

	def handle_end_of_active_category(self):

		unannotated_indices = self.get_indices_for_status('Un-annotated')
		if len(unannotated_indices) > 0:
			self.active_category = 'Un-annotated'
			self.annotations_complete_mode = False
			self.navigate_to_image_index(unannotated_indices[0])
			return
		ai_labeled_indices = self.get_indices_for_status('Annotator-labeled')
		if len(ai_labeled_indices) > 0:
			self.active_category = 'Annotator-labeled'
			self.annotations_complete_mode = False
			self.navigate_to_image_index(ai_labeled_indices[0])
			return
		self.active_category = None
		self.annotations_complete_mode = True
		self.flash_panel.Hide()
		self.refresh_status_bar()
		self.canvas.Refresh()

	def navigate_in_active_category(self, direction):

		if self.active_category is None or not self.image_paths:
			return False
		indices = self.get_indices_for_status(self.active_category)
		if len(indices) == 0:
			self.handle_end_of_active_category()
			return True

		current_idx = self.current_image_id
		if direction == 'next':
			for idx in indices:
				if idx > current_idx:
					self.annotations_complete_mode = False
					self.navigate_to_image_index(idx)
					return True
			if len(indices) == 1 and current_idx == indices[0]:
				return True
			self.handle_end_of_active_category()
			return True

		for idx in reversed(indices):
			if idx < current_idx:
				self.annotations_complete_mode = False
				self.navigate_to_image_index(idx)
				return True
		if len(indices) == 1 and current_idx == indices[0]:
			return True
		if current_idx != indices[0]:
			self.annotations_complete_mode = False
			self.navigate_to_image_index(indices[0])
		return True

	def refresh_status_bar(self):

		if not self.status_buttons:
			return

		counts = self.get_status_counts()
		current_status = None
		if self.image_paths and 0 <= self.current_image_id < len(self.image_paths):
			current_name = os.path.basename(self.image_paths[self.current_image_id])
			current_status = self.image_status.get(current_name, 'Un-annotated')

		for status, button in self.status_buttons.items():
			button.SetLabel('[ ' + self.status_chip_display_name(status) + ': ' + str(counts.get(status, 0)) + ' ]')
			is_current = status == current_status
			is_active = status == self.active_category
			if is_current and is_active:
				button.SetBackgroundColour(wx.Colour(90, 80, 190))
				button.SetForegroundColour(wx.Colour(255, 255, 255))
				font = button.GetFont()
				font.SetWeight(wx.FONTWEIGHT_BOLD)
				button.SetFont(font)
			elif is_current:
				button.SetBackgroundColour(wx.Colour(60, 130, 210))
				button.SetForegroundColour(wx.Colour(255, 255, 255))
				font = button.GetFont()
				font.SetWeight(wx.FONTWEIGHT_BOLD)
				button.SetFont(font)
			elif is_active:
				button.SetBackgroundColour(wx.Colour(245, 205, 80))
				button.SetForegroundColour(wx.Colour(40, 40, 40))
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
		self.update_annotation_presence_indicator()

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
		self.hide_flash_message()
		self.dismiss_annotations_complete()
		self.current_image_id = target_index
		self.load_current_image()
		self.canvas.SetFocus()
		if self.thumbnail_popup is not None:
			self.thumbnail_popup.Dismiss()
			self.thumbnail_popup = None

	def on_status_more_click(self, event, status):

		if self.thumbnail_popup is not None:
			self.thumbnail_popup.Dismiss()
			self.thumbnail_popup = None

		indices = self.get_indices_for_status(status)
		button = self.status_more_buttons.get(status)
		if button is None:
			self.canvas.SetFocus()
			return

		self.thumbnail_popup = FilenameListPopup(self, indices, self.image_paths, self.navigate_to_image_index, popup_width=300, max_height=400)
		self.thumbnail_popup.Position(button.ClientToScreen((0, button.GetSize().height)), (0, 0))
		self.thumbnail_popup.Popup()
		self.canvas.SetFocus()

	def on_status_chip_click(self, status):

		if not self.image_paths:
			return
		self.hide_flash_message()
		self.dismiss_annotations_complete()
		if self.active_category == status:
			self.active_category = None
			self.refresh_status_bar()
			self.canvas.Refresh()
			self.canvas.SetFocus()
			return
		self.active_category = status
		indices = self.get_indices_for_status(status)
		if len(indices) == 0:
			self.refresh_status_bar()
			self.canvas.Refresh()
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

		is_edit = self.mode == 'EDIT'
		if not is_edit:
			self.AI_help = False
			self.mode_indicator.SetLabel('● Review Mode')
			self.mode_indicator.SetForegroundColour(wx.Colour(90, 120, 170))
		else:
			self.mode_indicator.SetLabel('● Edit Mode')
			self.mode_indicator.SetForegroundColour(wx.Colour(60, 150, 80))
		self.Layout()
		self.canvas.SetFocus()


	def ensure_sam2_loaded(self):

		if self.sam2_loaded:
			return True
		if self.model_cp is None or self.model_cfg is None:
			wx.MessageBox(f'{self.sam_model_name} has not been set up. AI Help remains OFF.', 'AI assistance OFF', wx.ICON_INFORMATION)
			return False
		self.show_flash_message('Loading AI model...')
		self.sam2_last_error = None
		try:
			self.sam2 = self.sam2_model()
			self.sam2_loaded = True
		except Exception as load_error:
			self.sam2_last_error = load_error
			self.sam2_loaded = False
			self.hide_flash_message()
			return False
		self.hide_flash_message()
		return True


	def set_sam_image_if_needed(self):

		if self.mode != 'EDIT' or not self.AI_help:
			return
		if not self.ensure_sam2_loaded():
			self.AI_help = False
			return
		image = Image.open(self.image_paths[self.current_image_id])
		image = np.array(image.convert('RGB'))
		self.sam2.set_image(image)


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
			self.set_filename_label(image_name)
			if image_name not in self.information:
				self.information[image_name] = {'polygons': [], 'class_names': []}
			self.current_polygon = []
			self.foreground_points = []
			self.background_points = []
			self.zoom_scale = 1.0
			self.update_canvas_geometry()
			self.scrolled_canvas.Scroll(0, 0)
			if self.flash_message_image_name is not None and self.flash_message_image_name != image_name:
				self.hide_flash_message()
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

		self.hide_flash_message()
		self.dismiss_annotations_complete()
		if self.navigate_in_active_category('prev'):
			self.canvas.SetFocus()
			return
		if self.image_paths and self.current_image_id > 0:
			self.current_image_id -= 1
			self.load_current_image()
		self.canvas.SetFocus()


	def next_image(self, event):

		self.hide_flash_message()
		self.dismiss_annotations_complete()
		if self.navigate_in_active_category('next'):
			self.canvas.SetFocus()
			return
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
				self.set_filename_label('')
				self.canvas.Refresh()
				self.refresh_status_bar()
				return
			if self.current_image_id >= len(self.image_paths):
				self.current_image_id = len(self.image_paths) - 1
			self.load_current_image()
		self.canvas.SetFocus()


	def image_coords_from_event(self, event):

		if self.annotations_complete_mode:
			return None
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

		dc = wx.PaintDC(self.canvas)
		if self.annotations_complete_mode:
			self.flash_panel.Hide()
			dc.SetBackground(wx.Brush(wx.Colour(24, 24, 28)))
			dc.Clear()
			dc.SetTextForeground(wx.Colour(255, 255, 255))
			dc.SetFont(wx.Font(wx.FontInfo(36).Bold().FaceName('Arial')))
			message = 'All annotations complete.'
			text_w, text_h = dc.GetTextExtent(message)
			canvas_w, canvas_h = self.canvas.GetClientSize()
			dc.DrawText(message, max(10, int((canvas_w - text_w) / 2)), max(10, int((canvas_h - text_h) / 2)))
			return

		if self.current_image is None:
			return

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

		if self.annotations_complete_mode:
			return
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

		if self.annotations_complete_mode:
			return
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
					self.finalize_status_after_edit(image_name)
					self.refresh_status_bar()
					self.show_flash_message('Annotation removed. Press Next or stay to re-annotate.')

		self.canvas.Refresh()


	def on_key_press(self, event):

		key_code = event.GetKeyCode()
		if key_code in (ord('Z'), ord('z')) and (event.CmdDown() or event.ControlDown()):
			self.undo_last_action()
			return
		if key_code in (ord('A'), ord('a')):
			image_name = self.get_current_image_name()
			if image_name is not None and self.image_status.get(image_name) == 'Annotator-labeled':
				self.push_accept_undo_snapshot(image_name)
				self.image_status[image_name] = 'Annotated'
				self.refresh_status_bar()
				self.show_flash_message('Accepted. Press Next to continue.')
			return

		if key_code == wx.WXK_TAB:
			if self.mode == 'REVIEW':
				self.mode = 'EDIT'
				if self.ai_model_mode == 'Off':
					self.AI_help = False
				else:
					if self.sam2_loaded:
						print(f'{self.sam_model_name} already loaded, ready.')
					elif self.ai_model_mode == 'On':
						if self.ensure_sam2_loaded():
							print(f'{self.sam_model_name} loaded into memory.')
						else:
							err = self.sam2_last_error
							print(f'{self.sam_model_name} failed to load: {err}')
					elif self.ai_model_mode == '':
						print(f'AI Model (Auto): loading {self.sam_model_name} now...')
						if self.ensure_sam2_loaded():
							print(f'{self.sam_model_name} loaded into memory.')
						else:
							err = self.sam2_last_error
							print(f'{self.sam_model_name} failed to load: {err}')
					self.AI_help = self.sam2_loaded
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
				current_status = self.image_status.get(image_name)
				self.push_undo_snapshot('no_subject', image_name)
				if current_status == 'Annotator-labeled':
					if image_name not in self.information:
						self.information[image_name] = {'polygons': [], 'class_names': []}
					self.information[image_name]['polygons'] = []
					self.information[image_name]['class_names'] = []
					self.no_subject_overrides.add(image_name)
					self.image_status[image_name] = 'No Subject'
					self.refresh_status_bar()
					self.canvas.Refresh()
					self.show_flash_message('Marked as No Subject. Annotations cleared.')
				elif image_name in self.no_subject_overrides:
					self.no_subject_overrides.remove(image_name)
					self.image_status[image_name] = self.infer_status_from_current_polygons(image_name)
					self.refresh_status_bar()
					self.show_flash_message('No Subject removed. Press Next to continue.')
				else:
					self.no_subject_overrides.add(image_name)
					self.image_status[image_name] = 'No Subject'
					self.refresh_status_bar()
					self.show_flash_message('Marked as No Subject. Press Next to continue.')
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
				committed = False
				if dialog.ShowModal() == wx.ID_OK:
					self.current_classname = dialog.GetStringSelection()
					if len(self.current_polygon) > 0:
						self.push_undo_snapshot('commit', image_name)
						self.current_polygon.append(self.current_polygon[0])
						self.information[image_name]['polygons'].append(self.current_polygon)
						self.information[image_name]['class_names'].append(self.current_classname)
						committed = True
				dialog.Destroy()
				self.current_polygon = []
				self.foreground_points = []
				self.background_points = []
				self.finalize_status_after_edit(image_name)
				self.refresh_status_bar()
				if committed:
					self.show_flash_message('Annotation saved. Press Next to continue.')
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

		if self.annotations_complete_mode:
			return
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
			image_name = self.get_current_image_name()
			self.finalize_status_after_edit(image_name)
			self.refresh_status_bar()
		self.vertex_drag_snapshot = None
		self.vertex_drag_moved = False
		self.selected_point = None
		self.canvas.Refresh()


	def on_mousewheel(self, event):

		if self.annotations_complete_mode:
			return
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
		self._position_flash_panel()
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
