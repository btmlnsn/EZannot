import os
import cv2
import wx
import json
import random
import torch
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
from screeninfo import get_monitors
from EZannot.sam2.build_sam import build_sam2
from EZannot.sam2.sam2_image_predictor import SAM2ImagePredictor
from .annotator import AutoAnnotation
from .tools import read_annotation,mask_to_polygon,generate_annotation
from .gui_main import open_or_select_page
from .gui_merge import PanelLv2_MergeDatasets



the_absolute_current_path=str(Path(__file__).resolve().parent)

FILENAME_INFO_TEXT=(
	'If you are using images exported by LabGym, filenames look like:\n'
	'\n'
	'  trial01_2000.jpg\n'
	'  │       │\n'
	'  │       └─ frame number (count within the generation window)\n'
	'  └─ video name\n'
	'\n'
	'Most users generate frames from the beginning of the video to the end.\n'
	'In that case, start time is 0 and you can convert frame to time with:\n'
	'\n'
	'  time (seconds) ≈ frame ÷ fps\n'
	'\n'
	'Example (from the start, 30 fps): trial01_2000.jpg → about 66.7 s.\n'
	'\n'
	'If generation did not start at the beginning of the video, add the\n'
	'LabGym start time:\n'
	'\n'
	'  time (seconds) ≈ start_t + (frame ÷ fps)\n'
	'\n'
	'Example (start_t = 10 s, 30 fps): trial01_2000.jpg → about 76.7 s.\n'
	'\n'
	'EZannot may append tags after the frame number (ignore them when\n'
	'reading the frame index):\n'
	'  • augmentation: _rot…, _flph/_flpv, _brih/_bril, _exph/_expl, _blur\n'
	'  • measurement: _annotated\n'
	'  • tiles: _x_y\n'
	'\n'
	'Example: trial01_2000_rot3.jpg → still frame 2000.'
)



class ColorPicker(wx.Dialog):

	def __init__(self,parent,title,name_and_color):

		super(ColorPicker,self).__init__(parent=None,title=title,size=(200,200))

		self.name_and_color=name_and_color
		name=self.name_and_color[0]
		hex_color=self.name_and_color[1].lstrip('#')
		color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.color_picker=wx.ColourPickerCtrl(self,colour=color)

		button=wx.Button(self,wx.ID_OK,label='Apply')

		boxsizer.Add(0,10,0)
		boxsizer.Add(self.color_picker,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(button,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(0,10,0)

		self.SetSizer(boxsizer)



class PanelLv1_AnnotationModule(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.dispaly_window()


	def dispaly_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_manualannotate=wx.Button(panel,label='Annotate Manually',size=(300,40))
		button_manualannotate.Bind(wx.EVT_BUTTON,self.manual_annotate)
		wx.Button.SetToolTip(button_manualannotate,'Use AI assistance to manually annotate a small set of initial training images for training an Annotator or refine the automatic annotations performed by an Annotator.')
		boxsizer.Add(button_manualannotate,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_autoannotate=wx.Button(panel,label='Annotate Automatically',size=(300,40))
		button_autoannotate.Bind(wx.EVT_BUTTON,self.auto_annotate)
		wx.Button.SetToolTip(button_autoannotate,'Use a trained Annotator to automatically annotate selected images for you.')
		boxsizer.Add(button_autoannotate,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_mergedatasets=wx.Button(panel,label='Merge Datasets',size=(300,40))
		button_mergedatasets.Bind(wx.EVT_BUTTON,self.merge_datasets)
		wx.Button.SetToolTip(button_mergedatasets,'Combine two folders of images and annotations into one dataset. Useful for merging batch runs or manual and auto annotations.')
		boxsizer.Add(button_mergedatasets,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def manual_annotate(self,event):

		open_or_select_page(self.notebook,'Annotate Manually',lambda:PanelLv2_ManualAnnotation(self.notebook))


	def auto_annotate(self,event):

		open_or_select_page(self.notebook,'Annotate Automatically',lambda:PanelLv2_AutoAnnotation(self.notebook))


	def merge_datasets(self,event):

		open_or_select_page(self.notebook,'Merge Datasets',lambda:PanelLv2_MergeDatasets(self.notebook))



class PanelLv2_ManualAnnotation(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_images=None
		self.result_path=None
		self.model_cp=None
		self.model_cfg=None
		self.color_map={}
		self.aug_methods=[]

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the image(s)\nto annotate',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_input,'Select one or more images. Common image formats (jpg, png, tif) are supported. If there is an annotation file in the same folder, EZannot will read the annotation file and show all the existing annotations.')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe annotated images',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Copies of images (including augmented ones) and the annotation file will be stored in this folder. The annotation file for the original (unaugmented) images will be stored in the origianl image folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_model=wx.BoxSizer(wx.HORIZONTAL)
		button_model=wx.Button(panel,label='Set up the SAM2 model for\nAI-assisted annotation',size=(300,40))
		button_model.Bind(wx.EVT_BUTTON,self.select_model)
		wx.Button.SetToolTip(button_model,'Choose the SAM2 model. If select from a folder, make sure the folder stores a checkpoint (*.pt) file and a corresponding model config (*.yaml) file.')
		self.text_model=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_model.Add(button_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_model.Add(self.text_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_classes=wx.BoxSizer(wx.HORIZONTAL)
		button_classes=wx.Button(panel,label='Specify the object classes and\ntheir annotation colors',size=(300,40))
		button_classes.Bind(wx.EVT_BUTTON,self.specify_classes)
		wx.Button.SetToolTip(button_classes,'Enter the name of each class and specify its annotation color.')
		self.text_classes=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_classes.Add(button_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_classes.Add(self.text_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_augmentation=wx.BoxSizer(wx.HORIZONTAL)
		button_augmentation=wx.Button(panel,label='Specify the augmentation methods\nfor the annotated images',size=(300,40))
		button_augmentation.Bind(wx.EVT_BUTTON,self.specify_augmentation)
		wx.Button.SetToolTip(button_augmentation,
			'Augmentation can greatly enhance the training efficiency. But for the first time of annotating an image set, you can skip this to keep an unaugmented, origianl annotated image set and import it to EZannot later to perform augmentation.')
		self.text_augmentation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_augmentation.Add(button_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_augmentation.Add(self.text_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to annotate images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_annotation)
		wx.Button.SetToolTip(button_startannotation,'Manually annotate objects in images.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_startannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.tif;*.tiff'
		dialog=wx.FileDialog(self,'Select images(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			self.text_input.SetLabel('Select: '+str(len(self.path_to_images))+' images in'+str(path)+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('The annotated images will be in: '+self.result_path+'.')
		dialog.Destroy()


	def select_model(self,event):

		path_to_sam2_model=None
		sam2_model_path=os.path.join(the_absolute_current_path,'sam2 models')
		sam2_models=[i for i in os.listdir(sam2_model_path) if os.path.isdir(os.path.join(sam2_model_path,i))]
		if '__pycache__' in sam2_models:
			sam2_models.remove('__pycache__')
		if '__init__' in sam2_models:
			sam2_models.remove('__init__')
		if '__init__.py' in sam2_models:
			sam2_models.remove('__init__.py')
		sam2_models.sort()
		if 'Choose a new directory of the SAM2 model' not in sam2_models:
			sam2_models.append('Choose a new directory of the SAM2 model')

		dialog=wx.SingleChoiceDialog(self,message='Select a SAM2 model for AI-assisted annotation.',caption='Select a SAM2 model',choices=sam2_models)
		if dialog.ShowModal()==wx.ID_OK:
			sam2_model=dialog.GetStringSelection()
			if sam2_model=='Choose a new directory of the SAM2 model':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					path_to_sam2_model=dialog1.GetPath()
				else:
					path_to_sam2_model=None
				dialog1.Destroy()
			else:
				path_to_sam2_model=os.path.join(sam2_model_path,sam2_model)
		dialog.Destroy()

		if path_to_sam2_model is None:
			wx.MessageBox('No SAM2 model is set up. The AI assistance function is OFF.','AI assistance OFF',wx.ICON_INFORMATION)
			self.text_model.SetLabel('No SAM2 model is set up. The AI assistance function is OFF.')
		else:
			for i in os.listdir(path_to_sam2_model):
				if i.endswith('.pt') and i.split('sam')[0]!='._':
					self.model_cp=os.path.join(path_to_sam2_model,i)
				if i.endswith('.yaml') and i.split('sam')[0]!='._':
					self.model_cfg=os.path.join(path_to_sam2_model,i)
			if self.model_cp is None:
				self.text_model.SetLabel('Missing checkpoint file.')
			elif self.model_cfg is None:
				self.text_model.SetLabel('Missing config file.')
			else:
				self.text_model.SetLabel('Checkpoint: '+str(os.path.basename(self.model_cp))+'; Config: '+str(os.path.basename(self.model_cfg))+'.')


	def specify_classes(self,event):

		if self.path_to_images is None:

			wx.MessageBox('No input images(s).','Error',wx.OK|wx.ICON_ERROR)

		else:

			annotation_files=[]
			color_map={}
			self.color_map={}
			classnames=''
			entry=None
			for i in os.listdir(os.path.dirname(self.path_to_images[0])):
				if i.endswith('.json'):
					annotation_files.append(os.path.join(os.path.dirname(self.path_to_images[0]),i))

			if len(annotation_files)>0:
				for annotation_file in annotation_files:
					if os.path.exists(annotation_file):
						annotation=json.load(open(annotation_file))
						for i in annotation['categories']:
							if i['id']>0:
								classname=i['name']
								if classname not in classnames:
									classnames=classnames+classname+','
				classnames=classnames[:-1]
				dialog=wx.MessageDialog(self,'Current classnames are: '+classnames+'.\nDo you want to modify the classnames?','Modify classnames?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					dialog1=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names',value=classnames)
					if dialog1.ShowModal()==wx.ID_OK:
						entry=dialog1.GetValue()
					dialog1.Destroy()
				else:
					entry=classnames
				dialog.Destroy()
			else:
				dialog=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names')
				if dialog.ShowModal()==wx.ID_OK:
					entry=dialog.GetValue()
				dialog.Destroy()

			if entry:
				try:
					for i in entry.split(','):
						color_map[i]='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
				except:
					color_map={}
					wx.MessageBox('Please enter the object class names in\ncorrect format! For example: apple,orange,pear','Error',wx.OK|wx.ICON_ERROR)

			if len(color_map)>0:
				for classname in color_map:
					dialog=ColorPicker(self,str(classname),[classname,color_map[classname]])
					if dialog.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog.color_picker.GetColour()
						self.color_map[classname]=(r,b,g)
					else:
						self.color_map[classname]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
					dialog.Destroy()
				self.text_classes.SetLabel('Classname:color: '+str(self.color_map)+'.')
			else:
				self.text_classes.SetLabel('None.')


	def specify_augmentation(self,event):

		aug_methods=['random rotation','horizontal flipping','vertical flipping','random brightening','random dimming','random blurring']
		selected=''
		dialog=wx.MultiChoiceDialog(self,message='Data augmentation methods',caption='Augmentation methods',choices=aug_methods)
		if dialog.ShowModal()==wx.ID_OK:
			self.aug_methods=[aug_methods[i] for i in dialog.GetSelections()]
			for i in self.aug_methods:
				if selected=='':
					selected=selected+i
				else:
					selected=selected+','+i
		else:
			self.aug_methods=[]
			selected='none'
		dialog.Destroy()

		if len(self.aug_methods)<=0:
			selected='none'

		self.text_augmentation.SetLabel('Augmentation methods: '+selected+'.')	


	def start_annotation(self,event):

		if self.path_to_images is None or self.result_path is None or len(self.color_map)==0:
			wx.MessageBox('No input images(s) / output folder / class names.','Error',wx.OK|wx.ICON_ERROR)
		else:
			WindowLv3_AnnotateImages(None,'Manually Annotate Images',self.path_to_images,self.result_path,self.color_map,self.aug_methods,model_cp=self.model_cp,model_cfg=self.model_cfg)



class MiniSwitch(wx.Panel):
	"""Small on/off switch control (wx.Switch is unavailable on this wx build)."""

	def __init__(self,parent,value=False,size=(36,20)):

		super().__init__(parent,size=size)
		self._value=bool(value)
		self._enabled=True
		self.SetMinSize(size)
		self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
		self.Bind(wx.EVT_PAINT,self.on_paint)
		self.Bind(wx.EVT_LEFT_DOWN,self.on_click)
		self.Bind(wx.EVT_SIZE,self.on_size)


	def GetValue(self):

		return self._value


	def SetValue(self,value):

		self._value=bool(value)
		self.Refresh()


	def Enable(self,enable=True):

		self._enabled=bool(enable)
		super().Enable(enable)
		self.Refresh()


	def Disable(self):

		self.Enable(False)


	def IsEnabled(self):

		return self._enabled


	def on_size(self,event):

		self.Refresh()
		event.Skip()


	def on_click(self,event):

		if not self._enabled:
			return
		self._value=not self._value
		self.Refresh()
		evt=wx.CommandEvent(wx.wxEVT_COMMAND_CHECKBOX_CLICKED,self.GetId())
		evt.SetEventObject(self)
		evt.SetInt(int(self._value))
		self.ProcessEvent(evt)


	def on_paint(self,event):

		dc=wx.AutoBufferedPaintDC(self)
		w,h=self.GetClientSize()
		bg=self.GetParent().GetBackgroundColour()
		if not bg.IsOk():
			bg=wx.SystemSettings.GetColour(wx.SYS_COLOUR_FRAMEBK)
		dc.SetBackground(wx.Brush(bg))
		dc.Clear()
		pad=1
		track_h=max(12,h-2*pad)
		track_y=(h-track_h)//2
		radius=track_h/2
		if not self._enabled:
			track_color=wx.Colour(210,210,214)
			knob_fill=wx.Colour(245,245,245)
			knob_edge=wx.Colour(230,230,230)
		elif self._value:
			track_color=wx.Colour(52,199,89)  # mac-like green
			knob_fill=wx.Colour(255,255,255)
			knob_edge=wx.Colour(220,220,220)
		else:
			track_color=wx.Colour(174,174,178)
			knob_fill=wx.Colour(255,255,255)
			knob_edge=wx.Colour(220,220,220)
		dc.SetBrush(wx.Brush(track_color))
		dc.SetPen(wx.Pen(track_color))
		dc.DrawRoundedRectangle(pad,track_y,w-2*pad,track_h,radius)
		knob_d=track_h-2
		knob_y=track_y+1
		if self._value and self._enabled:
			knob_x=w-pad-knob_d-1
		else:
			knob_x=pad+1
		dc.SetBrush(wx.Brush(knob_fill))
		dc.SetPen(wx.Pen(knob_edge))
		dc.DrawCircle(knob_x+knob_d//2,knob_y+knob_d//2,knob_d//2)



class QueueListDialog(wx.Dialog):

	"""Scrollable, searchable list of images in one annotator queue."""

	def __init__(self,parent,queue_title,paths):

		super().__init__(parent,title=f'{queue_title} images',size=(480,520))
		self._all_paths=list(paths)
		self._visible_paths=[]
		self.selected_path=None

		root=wx.BoxSizer(wx.VERTICAL)

		self.search=wx.SearchCtrl(self,style=wx.TE_PROCESS_ENTER)
		self.search.SetDescriptiveText('Search filenames…')
		self.search.ShowCancelButton(True)
		self.search.Bind(wx.EVT_TEXT,self.on_filter)
		self.search.Bind(wx.EVT_SEARCHCTRL_CANCEL_BTN,self.on_clear_search)
		self.search.Bind(wx.EVT_TEXT_ENTER,self.on_activate)
		root.Add(self.search,0,wx.EXPAND|wx.ALL,10)

		self.listbox=wx.ListBox(self,style=wx.LB_SINGLE)
		self.listbox.Bind(wx.EVT_LISTBOX_DCLICK,self.on_activate)
		self.listbox.Bind(wx.EVT_KEY_DOWN,self.on_list_key)
		root.Add(self.listbox,1,wx.EXPAND|wx.LEFT|wx.RIGHT,10)

		self.status=wx.StaticText(self,label='')
		root.Add(self.status,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP,10)

		buttons=wx.BoxSizer(wx.HORIZONTAL)
		copy_btn=wx.Button(self,label='Copy all')
		copy_btn.Bind(wx.EVT_BUTTON,self.on_copy_all)
		wx.Button.SetToolTip(copy_btn,'Copy the filenames currently shown in the list')
		buttons.Add(copy_btn,0,wx.RIGHT,8)
		buttons.AddStretchSpacer(1)
		go_btn=wx.Button(self,label='Go to image')
		go_btn.Bind(wx.EVT_BUTTON,self.on_activate)
		cancel_btn=wx.Button(self,wx.ID_CANCEL,label='Close')
		buttons.Add(go_btn,0,wx.RIGHT,8)
		buttons.Add(cancel_btn,0)
		root.Add(buttons,0,wx.EXPAND|wx.ALL,10)

		self.SetSizer(root)
		self.refill('')
		self.search.SetFocus()


	def on_clear_search(self,event):

		self.search.ChangeValue('')
		self.refill('')


	def on_filter(self,event):

		self.refill(self.search.GetValue())


	def refill(self,query):

		needle=query.strip().lower()
		self.listbox.Clear()
		self._visible_paths=[]
		for path in self._all_paths:
			name=os.path.basename(path)
			if needle and needle not in name.lower():
				continue
			self._visible_paths.append(path)
			self.listbox.Append(name)
		total=len(self._all_paths)
		shown=len(self._visible_paths)
		if needle:
			self.status.SetLabel(f'Showing {shown} of {total}')
		else:
			self.status.SetLabel(f'{shown} image{"s" if shown!=1 else ""}')


	def on_list_key(self,event):

		if event.GetKeyCode() in (wx.WXK_RETURN,wx.WXK_NUMPAD_ENTER):
			self.on_activate(event)
		else:
			event.Skip()


	def on_activate(self,event):

		index=self.listbox.GetSelection()
		if index==wx.NOT_FOUND or index<0 or index>=len(self._visible_paths):
			return
		self.selected_path=self._visible_paths[index]
		self.EndModal(wx.ID_OK)


	def on_copy_all(self,event):

		lines=[os.path.basename(p) for p in self._visible_paths]
		text='\n'.join(lines)
		if wx.TheClipboard.Open():
			wx.TheClipboard.SetData(wx.TextDataObject(text))
			wx.TheClipboard.Close()



class WindowLv3_AnnotateImages(wx.Frame):

	QUEUE_ORDER=('pending','annotated','skipped')
	QUEUE_TITLES={'pending':'Pending','annotated':'Annotated','skipped':'Skipped'}

	def __init__(self,parent,title,path_to_images,result_path,color_map,aug_methods,model_cp=None,model_cfg=None):

		monitor=get_monitors()[0]
		# Leave room below the menu bar (y=50) and above the dock / screen edge.
		super().__init__(parent,title=title,pos=(10,50),size=(monitor.width-20,monitor.height-120))

		self.image_paths=path_to_images
		self.result_path=result_path
		self.color_map=color_map
		self.aug_methods=aug_methods
		self.model_cp=model_cp
		self.model_cfg=model_cfg
		self.current_image_id=0
		self.current_image=None
		self.current_segmentation=None
		self.current_polygon=[]
		self.current_classname=list(self.color_map.keys())[0]
		self.information=read_annotation(os.path.dirname(self.image_paths[0]),color_map=self.color_map)
		# Empty COCO image rows (no polygons) were previously exported as skips.
		self.skipped_images={
			name for name,info in self.information.items()
			if len(info.get('polygons',[]))==0
			}
		self.foreground_points=[]
		self.background_points=[]
		self.selected_point=None
		self.start_modify=False
		self.show_name=False
		self.AI_help=False
		self.scale=1.0
		self.min_scale=0.25
		self.max_scale=8.0
		self.zoom_step=1.25
		self.active_queue='pending'
		self._queue_last_path={q:None for q in self.QUEUE_ORDER}

		self.init_ui()
		self.active_queue=self._default_queue()
		start_paths=[p for p in self.image_paths if self._structural_class(os.path.basename(p))==self.active_queue]
		if start_paths:
			self.current_image_id=self.image_paths.index(start_paths[0])
		self.load_current_image()


	def sam2_model(self):

		device='cuda' if torch.cuda.is_available() else 'cpu'
		predictor=SAM2ImagePredictor(build_sam2(self.model_cfg,self.model_cp,device=device))
		return predictor


	def init_ui(self):

		panel=wx.Panel(self)
		self.ui_panel=panel
		vbox=wx.BoxSizer(wx.VERTICAL)
		hbox=wx.BoxSizer(wx.HORIZONTAL)

		self.ai_button=wx.ToggleButton(panel,label='AI Help: OFF',size=(200,30))
		self.ai_button.Bind(wx.EVT_TOGGLEBUTTON,self.toggle_ai)
		hbox.Add(self.ai_button,flag=wx.ALL,border=2)

		self.prev_button=wx.Button(panel,label='← Prev',size=(200,30))
		self.prev_button.Bind(wx.EVT_BUTTON,self.previous_image)
		hbox.Add(self.prev_button,flag=wx.ALL,border=2)

		self.next_button=wx.Button(panel,label='Next →',size=(200,30))
		self.next_button.Bind(wx.EVT_BUTTON,self.next_image)
		hbox.Add(self.next_button,flag=wx.ALL,border=2)

		self.delete_button=wx.Button(panel,label='Delete',size=(200,30))
		self.delete_button.Bind(wx.EVT_BUTTON,self.delete_image)
		hbox.Add(self.delete_button,flag=wx.ALL,border=2)

		self.export_button=wx.Button(panel,label='Export Annotations',size=(200,30))
		self.export_button.Bind(wx.EVT_BUTTON,self.export_annotations)
		hbox.Add(self.export_button,flag=wx.ALL,border=2)
		vbox.Add(hbox,flag=wx.ALIGN_CENTER|wx.TOP,border=5)

		queue_box=wx.BoxSizer(wx.HORIZONTAL)
		self.queue_buttons={}
		self.queue_list_buttons={}
		for i,queue_id in enumerate(self.QUEUE_ORDER):
			# Tight [toggle ☰] pair — plain glyph, no button chrome.
			pair=wx.BoxSizer(wx.HORIZONTAL)
			btn=wx.ToggleButton(panel,label=f'{self.QUEUE_TITLES[queue_id]} (0)',size=(118,26))
			btn.Bind(wx.EVT_TOGGLEBUTTON,lambda event,q=queue_id:self.on_queue_select(q))
			pair.Add(btn,0,wx.ALIGN_CENTER_VERTICAL)
			self.queue_buttons[queue_id]=btn
			list_icon=wx.StaticText(panel,label='☰')
			list_icon.SetCursor(wx.Cursor(wx.CURSOR_HAND))
			list_icon.Bind(wx.EVT_LEFT_DOWN,lambda event,q=queue_id:self.show_queue_list(q))
			wx.Window.SetToolTip(list_icon,f'Browse / search {self.QUEUE_TITLES[queue_id]} images')
			pair.Add(list_icon,0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,5)
			self.queue_list_buttons[queue_id]=list_icon
			queue_box.Add(pair,0,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,0 if i==0 else 10)
		vbox.Add(queue_box,flag=wx.ALIGN_CENTER|wx.TOP,border=4)

		# Single-height row 2: centered filename; switch pinned under Export.
		self.row2_panel=wx.Panel(panel)
		self.row2_panel.SetMinSize((-1,24))
		row2_sizer=wx.BoxSizer(wx.HORIZONTAL)
		row2_sizer.AddStretchSpacer(1)

		filename_inner=wx.BoxSizer(wx.HORIZONTAL)
		self.filename_info_button=wx.Button(self.row2_panel,label='i',size=(22,22))
		self.filename_info_button.Bind(wx.EVT_BUTTON,self.show_filename_info)
		wx.Button.SetToolTip(self.filename_info_button,'About this filename')
		filename_inner.Add(self.filename_info_button,0,wx.ALIGN_CENTER_VERTICAL|wx.RIGHT,6)

		self.text_filename=wx.StaticText(self.row2_panel,label='',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_MIDDLE)
		self.text_filename.SetMinSize((320,-1))
		self.text_filename.SetMaxSize((560,-1))
		filename_inner.Add(self.text_filename,0,wx.ALIGN_CENTER_VERTICAL)

		row2_sizer.Add(filename_inner,0,wx.ALIGN_CENTER_VERTICAL)
		row2_sizer.AddStretchSpacer(1)
		self.row2_panel.SetSizer(row2_sizer)
		vbox.Add(self.row2_panel,0,wx.EXPAND|wx.TOP|wx.BOTTOM,1)

		self.export_only_to_export_folder=False
		self.export_only_label=wx.StaticText(self.row2_panel,label='JSON in Export Folder only')
		label_font=self.text_filename.GetFont()
		if label_font.IsOk() and label_font.GetPointSize()>1:
			label_font=wx.Font(label_font)
			label_font.SetPointSize(max(label_font.GetPointSize()-1,9))
			self.export_only_label.SetFont(label_font)
		else:
			self.export_only_label.SetFont(wx.Font(wx.FontInfo(11)))
		self.export_only_switch=MiniSwitch(self.row2_panel,value=False,size=(34,18))
		self.export_only_switch.Bind(wx.EVT_CHECKBOX,self.toggle_export_only)
		tip=(
			'OFF (default): also write annotations.json into the original image folder.\n'
			'ON: write annotations.json only into the export folder.'
		)
		wx.Window.SetToolTip(self.export_only_switch,tip)
		wx.Window.SetToolTip(self.export_only_label,tip)
		self._export_only_tip=tip
		self.update_export_only_lock()

		self.scrolled_canvas=wx.ScrolledWindow(panel,style=wx.VSCROLL|wx.HSCROLL)
		self.scrolled_canvas.SetScrollRate(10,10)
		self.canvas=wx.Panel(self.scrolled_canvas,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-120))
		self.scrolled_canvas.SetBackgroundColour('black')

		self.canvas.Bind(wx.EVT_PAINT,self.on_paint)
		self.canvas.Bind(wx.EVT_LEFT_DOWN,self.on_left_click)
		self.canvas.Bind(wx.EVT_RIGHT_DOWN,self.on_right_click)
		self.canvas.Bind(wx.EVT_MOTION,self.on_left_move)
		self.canvas.Bind(wx.EVT_LEFT_UP,self.on_left_up)
		self.canvas.Bind(wx.EVT_MOUSEWHEEL,self.on_mousewheel)

		self.scrolled_canvas.SetSizer(wx.BoxSizer(wx.VERTICAL))
		self.scrolled_canvas.GetSizer().Add(self.canvas,proportion=1,flag=wx.EXPAND|wx.ALL,border=5)
		vbox.Add(self.scrolled_canvas,proportion=1,flag=wx.EXPAND|wx.ALL,border=5)

		panel.SetSizer(vbox)
		panel.Bind(wx.EVT_SIZE,self.on_ui_panel_size)
		self.row2_panel.Bind(wx.EVT_SIZE,self.on_ui_panel_size)
		self.Bind(wx.EVT_CHAR_HOOK,self.on_key_press)
		self.Show()
		wx.CallAfter(self.reposition_export_only_controls)


	def on_ui_panel_size(self,event):

		wx.CallAfter(self.reposition_export_only_controls)
		event.Skip()


	def reposition_export_only_controls(self):

		if not hasattr(self,'export_button') or not hasattr(self,'row2_panel'):
			return
		# Align label + switch on row 2, centered under Export Annotations.
		btn_pt=self.export_button.ClientToScreen((0,0))
		row_pt=self.row2_panel.ClientToScreen((0,0))
		local_x=btn_pt.x-row_pt.x
		bw=self.export_button.GetSize()[0]
		rh=self.row2_panel.GetClientSize()[1]
		lw,lh=self.export_only_label.GetBestSize()
		sw,sh=self.export_only_switch.GetSize()
		gap=6
		total_w=lw+gap+sw
		x=local_x+(bw-total_w)//2
		y=max(0,(rh-max(lh,sh))//2)
		self.export_only_label.SetPosition((x,y+(max(lh,sh)-lh)//2))
		self.export_only_switch.SetPosition((x+lw+gap,y+(max(lh,sh)-sh)//2))
		self.export_only_label.Raise()
		self.export_only_switch.Raise()


	def toggle_ai(self,event):

		if self.model_cp is None or self.model_cfg is None:

			self.ai_button.SetLabel('AI Help: OFF')
			wx.MessageBox('SAM2 model has not been set up.','Error',wx.ICON_ERROR)

		else:

			self.AI_help=self.ai_button.GetValue()
			if self.AI_help:
				self.ai_button.SetLabel('AI Help: ON')
			else:
				self.ai_button.SetLabel('AI Help: OFF')

			if self.AI_help:
				image=Image.open(self.image_paths[self.current_image_id])
				image=np.array(image.convert('RGB'))
				self.sam2=self.sam2_model()
				self.sam2.set_image(image)

		self.canvas.SetFocus()


	def toggle_export_only(self,event):

		if not self.export_only_switch.IsEnabled():
			self.export_only_switch.SetValue(False)
			self.export_only_to_export_folder=False
			self.canvas.SetFocus()
			return
		self.export_only_to_export_folder=self.export_only_switch.GetValue()
		self.canvas.SetFocus()


	def same_import_export_folder(self):

		if not self.image_paths or not self.result_path:
			return False
		try:
			return os.path.realpath(os.path.dirname(self.image_paths[0]))==os.path.realpath(self.result_path)
		except OSError:
			return os.path.normpath(os.path.dirname(self.image_paths[0]))==os.path.normpath(self.result_path)


	def update_export_only_lock(self):

		if not hasattr(self,'export_only_switch'):
			return
		locked=self.same_import_export_folder()
		if locked:
			self.export_only_switch.SetValue(False)
			self.export_only_to_export_folder=False
			self.export_only_switch.Disable()
			lock_tip='Unavailable when the import folder and export folder are the same.'
			wx.Window.SetToolTip(self.export_only_switch,lock_tip)
			wx.Window.SetToolTip(self.export_only_label,lock_tip)
			self.export_only_label.Enable(False)
		else:
			self.export_only_switch.Enable(True)
			self.export_only_label.Enable(True)
			wx.Window.SetToolTip(self.export_only_switch,self._export_only_tip)
			wx.Window.SetToolTip(self.export_only_label,self._export_only_tip)


	def current_image_name(self):

		if not self.image_paths:
			return None
		if self.current_image_id<0 or self.current_image_id>=len(self.image_paths):
			return None
		return os.path.basename(self.image_paths[self.current_image_id])


	def _polygon_count(self,name):

		info=self.information.get(name)
		if not info:
			return 0
		return len(info.get('polygons',[]))


	def _ensure_image_info(self,name):

		if name not in self.information:
			self.information[name]={'polygons':[],'class_names':[]}
		return self.information[name]


	def _image_record(self,name):

		return self.information.get(name,{'polygons':[],'class_names':[]})


	def _drop_empty_image_info(self,name):

		"""Clear empty state so the image is Pending (not Skipped)."""

		self.skipped_images.discard(name)
		if name in self.information and self._polygon_count(name)==0:
			del self.information[name]


	def _mark_image_skipped(self,name):

		"""Mark skipped only via Next / Export Skip; Export writes an empty COCO image row."""

		if name is None or self._polygon_count(name)>0:
			return
		self.skipped_images.add(name)
		self._ensure_image_info(name)


	def _current_is_blank(self):

		name=self.current_image_name()
		if name is None or self.current_image is None:
			return False
		return self._polygon_count(name)==0


	def _structural_class(self,name):

		n_poly=self._polygon_count(name)
		if n_poly>0:
			return 'annotated'
		if name in self.skipped_images:
			return 'skipped'
		return 'pending'


	def classify_image(self,name):

		"""Queue class for live counts (same rules as navigation)."""

		return self._structural_class(name)


	def _in_nav_queue(self,name,queue):

		"""Membership for Prev/Next within a queue."""

		return self._structural_class(name)==queue


	def queue_paths(self,queue=None):

		queue=self.active_queue if queue is None else queue
		return [p for p in self.image_paths if self._in_nav_queue(os.path.basename(p),queue)]


	def queue_counts(self):

		counts={q:0 for q in self.QUEUE_ORDER}
		for path in self.image_paths:
			counts[self.classify_image(os.path.basename(path))]+=1
		return counts


	def _default_queue(self):

		counts={q:0 for q in self.QUEUE_ORDER}
		for path in self.image_paths:
			counts[self._structural_class(os.path.basename(path))]+=1
		for queue_id in self.QUEUE_ORDER:
			if counts[queue_id]>0:
				return queue_id
		return 'pending'


	def refresh_queue_ui(self):

		if not hasattr(self,'queue_buttons'):
			return
		counts=self.queue_counts()
		for queue_id,btn in self.queue_buttons.items():
			btn.SetLabel(f'{self.QUEUE_TITLES[queue_id]} ({counts[queue_id]})')
			btn.SetValue(queue_id==self.active_queue)
		# Keep Next available for a blank image at end-of-queue so Next can still skip it.
		can_prev=self._prev_path_in_active_queue() is not None
		can_next=(
			self._next_path_in_active_queue() is not None
			or (self.current_image is not None and self._current_is_blank())
		)
		self.prev_button.Enable(can_prev)
		self.next_button.Enable(can_next)
		self.update_filename_label()


	def on_queue_select(self,queue_id):

		if queue_id==self.active_queue:
			self.queue_buttons[queue_id].SetValue(True)
			self.canvas.SetFocus()
			return
		if self.image_paths and 0<=self.current_image_id<len(self.image_paths):
			self._queue_last_path[self.active_queue]=self.image_paths[self.current_image_id]
		self.active_queue=queue_id
		paths=self.queue_paths()
		last=self._queue_last_path.get(queue_id)
		if last in paths:
			self.current_image_id=self.image_paths.index(last)
			self.load_current_image()
		elif paths:
			self.current_image_id=self.image_paths.index(paths[0])
			self.load_current_image()
		else:
			self.refresh_queue_ui()
		self.canvas.SetFocus()


	def show_queue_list(self,queue_id):

		title=self.QUEUE_TITLES[queue_id]
		paths=self.queue_paths(queue_id)
		dialog=QueueListDialog(self,title,paths)
		if dialog.ShowModal()==wx.ID_OK and dialog.selected_path:
			self.go_to_queue_path(queue_id,dialog.selected_path)
		dialog.Destroy()
		self.canvas.SetFocus()


	def go_to_queue_path(self,queue_id,path):

		"""Open a path from a queue list without treating the open as a skip."""

		if path not in self.image_paths:
			return
		if self.image_paths and 0<=self.current_image_id<len(self.image_paths):
			self._queue_last_path[self.active_queue]=self.image_paths[self.current_image_id]
		self.active_queue=queue_id
		self.current_image_id=self.image_paths.index(path)
		self.load_current_image()


	def _next_path_in_active_queue(self):

		"""Next path in the active queue after the current image (session order)."""

		if not self.image_paths:
			return None
		path=self.image_paths[self.current_image_id]
		paths=self.queue_paths()
		if path in paths:
			index=paths.index(path)
			if index<len(paths)-1:
				return paths[index+1]
			return None
		# Current left the active queue (annotated / cleared) — keep walking that queue.
		for candidate in self.image_paths[self.current_image_id+1:]:
			if self._in_nav_queue(os.path.basename(candidate),self.active_queue):
				return candidate
		return None


	def _prev_path_in_active_queue(self):

		"""Previous path in the active queue before the current image (session order)."""

		if not self.image_paths:
			return None
		path=self.image_paths[self.current_image_id]
		paths=self.queue_paths()
		if path in paths:
			index=paths.index(path)
			if index>0:
				return paths[index-1]
			return None
		for candidate in reversed(self.image_paths[:self.current_image_id]):
			if self._in_nav_queue(os.path.basename(candidate),self.active_queue):
				return candidate
		return None


	def _after_polygons_changed(self):

		# Stay on this image; Next advances within the queue the user was browsing.
		self.refresh_queue_ui()


	def update_filename_label(self):

		title=self.QUEUE_TITLES.get(self.active_queue,'Pending')
		if not self.image_paths:
			self.text_filename.SetLabel('Filename: No images')
			return
		if self.current_image is None or self.current_image_id<0 or self.current_image_id>=len(self.image_paths):
			self.text_filename.SetLabel(f'No {title.lower()} images')
			return
		path=self.image_paths[self.current_image_id]
		name=os.path.basename(path)
		paths=self.queue_paths()
		if path in paths:
			index=paths.index(path)+1
			self.text_filename.SetLabel(f'Filename: {name} ({index} / {len(paths)}) · {title}')
		else:
			# Still viewing an image that just changed class; Next continues this queue.
			self.text_filename.SetLabel(f'Filename: {name} · {title}')

	def show_filename_info(self,event):

		dialog=wx.Dialog(self,title='About this filename')
		sizer=wx.BoxSizer(wx.VERTICAL)
		text=wx.StaticText(dialog,label=FILENAME_INFO_TEXT)
		text.Wrap(360)
		sizer.Add(text,0,wx.ALL,16)
		ok_button=wx.Button(dialog,wx.ID_OK,label='OK')
		sizer.Add(ok_button,0,wx.ALIGN_RIGHT|wx.RIGHT|wx.BOTTOM,16)
		dialog.SetSizer(sizer)
		dialog.Fit()
		dialog.CentreOnParent()
		dialog.ShowModal()
		dialog.Destroy()
		self.canvas.SetFocus()


	def fit_image_to_view(self):

		if self.current_image is None:
			return

		img_width,img_height=self.current_image.GetSize()
		# Drop previous virtual size so scrollbars clear and client size is the full view for this image.
		self.scrolled_canvas.SetVirtualSize((1,1))
		self.scrolled_canvas.Layout()
		cw,ch=self.scrolled_canvas.GetClientSize()
		if cw>0 and ch>0 and img_width>0 and img_height>0:
			self.scale=max(min(cw/img_width,ch/img_height,1.0),self.min_scale)
		else:
			self.scale=1.0
		new_w=int(img_width*self.scale)
		new_h=int(img_height*self.scale)
		self.scrolled_canvas.SetVirtualSize((new_w,new_h))
		self.canvas.SetSize((new_w,new_h))
		self.scrolled_canvas.Scroll(0,0)
		self.canvas.Refresh()


	def load_current_image(self):

		if not self.image_paths:
			self.current_image=None
			self.refresh_queue_ui()
			self.canvas.Refresh()
			return

		if self.current_image_id>=len(self.image_paths):
			self.current_image_id=len(self.image_paths)-1
		if self.current_image_id<0:
			self.current_image_id=0

		path=self.image_paths[self.current_image_id]
		self.current_image=wx.Image(path,wx.BITMAP_TYPE_ANY)
		# Do not write into information on open — blank opens stay Pending until Next (skip)
		# or an annotation is committed.
		self.current_polygon=[]
		self.foreground_points=[]
		self.background_points=[]
		self.fit_image_to_view()
		# First open can happen before layout; re-fit once the scroll view has a real size.
		cw,ch=self.scrolled_canvas.GetClientSize()
		if cw<=0 or ch<=0:
			wx.CallAfter(self.fit_image_to_view)

		if self.AI_help:
			image=Image.open(path)
			image=np.array(image.convert('RGB'))
			self.sam2=self.sam2_model()
			self.sam2.set_image(image)

		self._queue_last_path[self.active_queue]=path
		self.refresh_queue_ui()


	def previous_image(self,event):

		# Prev does not skip — a blank open stays Pending.
		prev_path=self._prev_path_in_active_queue()
		if prev_path is not None:
			self.current_image_id=self.image_paths.index(prev_path)
			self.load_current_image()
		self.canvas.SetFocus()


	def next_image(self,event):

		if not self.image_paths or self.current_image is None:
			self.canvas.SetFocus()
			return
		path=self.image_paths[self.current_image_id]
		name=os.path.basename(path)
		next_path=self._next_path_in_active_queue()
		# Next without annotations is the skip action (even on the last image).
		if self._polygon_count(name)==0:
			self._mark_image_skipped(name)
		if next_path is not None:
			self.current_image_id=self.image_paths.index(next_path)
			self.load_current_image()
		else:
			self.refresh_queue_ui()
		self.canvas.SetFocus()


	def delete_image(self,event):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			queue_before=self.queue_paths()
			qi=queue_before.index(path) if path in queue_before else 0
			self.image_paths.remove(path)
			image_name=os.path.basename(path)
			self.skipped_images.discard(image_name)
			if image_name in self.information:
				del self.information[image_name]
			remaining=[p for p in queue_before if p!=path]
			if remaining:
				if qi>0:
					target=remaining[qi-1]
				else:
					target=remaining[0]
				self.current_image_id=self.image_paths.index(target)
				self.load_current_image()
			else:
				# Active queue empty (and/or session empty): do not open another queue's image.
				if self.current_image_id>=len(self.image_paths):
					self.current_image_id=max(len(self.image_paths)-1,0)
				self.current_image=None
				self.refresh_queue_ui()
				self.canvas.Refresh()
		self.canvas.SetFocus()


	def on_paint(self,event):

		if self.current_image is None:
			return

		dc=wx.PaintDC(self.canvas)
		w,h=self.current_image.GetSize()
		scaled_image=self.current_image.Scale(int(w*self.scale),int(h*self.scale),wx.IMAGE_QUALITY_HIGH)
		dc.DrawBitmap(wx.Bitmap(scaled_image),0,0,True)
		image_name=os.path.basename(self.image_paths[self.current_image_id])
		record=self._image_record(image_name)
		polygons=record['polygons']
		class_names=record['class_names']

		if len(polygons)>0:
			for i,polygon in enumerate(polygons):
				color=self.color_map[class_names[i]]
				pen=wx.Pen(wx.Colour(*color),width=2)
				dc.SetPen(pen)
				dc.DrawLines([(int(x*self.scale),int(y*self.scale)) for x,y in polygon])
				if self.start_modify:
					brush=wx.Brush(wx.Colour(*color))
					dc.SetBrush(brush)
					for x,y in polygon:
						dc.DrawCircle(int(x*self.scale),int(y*self.scale),4)
				if self.show_name:
					x_max=int(max(x for x,y in polygon)*self.scale)
					x_min=int(min(x for x,y in polygon)*self.scale)
					y_max=int(max(y for x,y in polygon)*self.scale)
					y_min=int(min(y for x,y in polygon)*self.scale)
					cx=int((x_max+x_min)/2)
					cy=int((y_max+y_min)/2)
					dc.SetTextForeground(wx.Colour(*color))
					dc.SetFont(wx.Font(wx.FontInfo(15).FaceName('Arial')))
					dc.DrawText(str(class_names[i]),cx,cy)

		if len(self.current_polygon)>0:
			current_polygon=[i for i in self.current_polygon]
			current_polygon.append(current_polygon[0])
			color=self.color_map[self.current_classname]
			brush=wx.Brush(wx.Colour(*color))
			dc.SetBrush(brush)
			for x,y in current_polygon:
				dc.DrawCircle(int(x*self.scale),int(y*self.scale),4)
			pen=wx.Pen(wx.Colour(*color),width=2)
			dc.SetPen(pen)
			dc.DrawLines([(int(x*self.scale),int(y*self.scale)) for x,y in current_polygon])


	def on_left_click(self,event):

		x,y=event.GetX(),event.GetY()

		if self.start_modify:

			image_name=os.path.basename(self.image_paths[self.current_image_id])
			for i,polygon in enumerate(self._image_record(image_name)['polygons']):
				for j,(px,py) in enumerate(polygon):
					if abs(px-int(x/self.scale))<5/self.scale and abs(py-int(y/self.scale))<5/self.scale:
						self.selected_point=(polygon,j,i)
						return

		else:

			if self.AI_help:
				self.foreground_points.append([int(x/self.scale),int(y/self.scale)])
				points=self.foreground_points+self.background_points
				labels=[1 for i in range(len(self.foreground_points))]+[0 for i in range(len(self.background_points))]
				masks,scores,logits=self.sam2.predict(point_coords=np.array(points),point_labels=np.array(labels))
				mask=masks[np.argsort(scores)[::-1]][0]
				self.current_polygon=mask_to_polygon(mask)
			else:
				self.current_polygon.append((int(x/self.scale),int(y/self.scale)))

		self.canvas.Refresh()


	def on_right_click(self,event):

		x,y=event.GetX(),event.GetY()

		if self.start_modify:

			return

		else:

			if len(self.current_polygon)>0:

				if self.AI_help:
					self.background_points.append([int(x/self.scale),int(y/self.scale)])
					points=self.foreground_points+self.background_points
					labels=[1 for i in range(len(self.foreground_points))]+[0 for i in range(len(self.background_points))]
					masks,scores,logits=self.sam2.predict(point_coords=np.array(points),point_labels=np.array(labels))
					mask=masks[np.argsort(scores)[::-1]][0]
					self.current_polygon=mask_to_polygon(mask)
				else:
					self.current_polygon.pop()

			else:

				to_delete=[]
				image_name=os.path.basename(self.image_paths[self.current_image_id])
				if image_name not in self.information:
					self.canvas.Refresh()
					return
				polygons=self.information[image_name]['polygons']
				if len(polygons)>0:
					for i,polygon in enumerate(polygons):
						x_max=max(x for x,y in polygon)
						x_min=min(x for x,y in polygon)
						y_max=max(y for x,y in polygon)
						y_min=min(y for x,y in polygon)
						if x_min<=int(x/self.scale)<=x_max and y_min<=int(y/self.scale)<=y_max:
							to_delete.append(i)
				if len(to_delete)>0:
					for i in sorted(to_delete,reverse=True):
						del self.information[image_name]['polygons'][i]
						del self.information[image_name]['class_names'][i]
					self._drop_empty_image_info(image_name)
					self._after_polygons_changed()

		self.canvas.Refresh()


	def on_key_press(self,event):

		key_code=event.GetKeyCode()

		if event.GetKeyCode()==wx.WXK_RETURN:
			if len(self.current_polygon)>2:
				classnames=sorted(list(self.color_map.keys()))
				current_index=classnames.index(self.current_classname)
				dialog=wx.SingleChoiceDialog(self,message='Choose object class name',caption='Class Name',choices=classnames)
				dialog.SetSelection(current_index)
				committed=False
				if dialog.ShowModal()==wx.ID_OK:
					self.current_classname=dialog.GetStringSelection()
					if len(self.current_polygon)>0:
						self.current_polygon.append(self.current_polygon[0])
						image_name=os.path.basename(self.image_paths[self.current_image_id])
						record=self._ensure_image_info(image_name)
						record['polygons'].append(self.current_polygon)
						record['class_names'].append(self.current_classname)
						self.skipped_images.discard(image_name)
						committed=True
				dialog.Destroy()
				self.current_polygon=[]
				self.foreground_points=[]
				self.background_points=[]
				self.canvas.Refresh()
				if committed:
					self._after_polygons_changed()
		elif key_code==wx.WXK_LEFT:
			self.previous_image(None)
		elif key_code==wx.WXK_RIGHT:
			self.next_image(None)
		elif event.GetKeyCode()==wx.WXK_SHIFT:
			if self.start_modify:
				self.start_modify=False
			else:
				self.start_modify=True
			self.canvas.Refresh()
		elif event.GetKeyCode()==wx.WXK_SPACE:
			if self.show_name:
				self.show_name=False
			else:
				self.show_name=True
			self.canvas.Refresh()
		elif event.GetKeyCode()==wx.WXK_ESCAPE:
			self.current_polygon=[]
			self.foreground_points=[]
			self.background_points=[]
			self.canvas.Refresh()
		else:
			event.Skip()


	def on_left_move(self,event):

		if self.selected_point is not None and event.Dragging() and event.LeftIsDown():
			polygon,j,i=self.selected_point
			x,y=event.GetX(),event.GetY()
			polygon[j]=(int(x/self.scale),int(y/self.scale))
			image_name=os.path.basename(self.image_paths[self.current_image_id])
			self.information[image_name]['polygons'][i]=polygon


	def on_left_up(self,event):
			
		self.selected_point=None
		self.canvas.Refresh()


	def on_mousewheel(self,event):

		if self.current_image is None:
			return

		if self.start_modify:
			return

		rotation=event.GetWheelRotation()
		if rotation>0:
			self.scale=min(self.scale*self.zoom_step,self.max_scale)
		else:
			self.scale=max(self.scale/self.zoom_step,self.min_scale)

		new_w=int(self.current_image.GetWidth()*self.scale)
		new_h=int(self.current_image.GetHeight()*self.scale)
		self.scrolled_canvas.SetVirtualSize((new_w,new_h))
		self.canvas.Refresh()


	def _resolve_blank_current_for_export(self):

		"""Ask whether the open blank Pending image is a Skip or should stay Pending."""

		if not self._current_is_blank():
			return True
		name=self.current_image_name()
		if name in self.skipped_images:
			return True
		dialog=wx.MessageDialog(
			self,
			f'Current image "{name}" has no annotations.\n\n'
			'Skip it (save as empty) or leave it pending for later?',
			'Current image unannotated',
			wx.YES_NO|wx.CANCEL|wx.ICON_QUESTION
		)
		dialog.SetYesNoCancelLabels('Skip image','Leave for later','Cancel')
		choice=dialog.ShowModal()
		dialog.Destroy()
		if choice==wx.ID_CANCEL:
			return False
		if choice==wx.ID_YES:
			self._mark_image_skipped(name)
		else:
			self._drop_empty_image_info(name)
		self.refresh_queue_ui()
		return True


	def export_annotations(self,event):

		if not self._resolve_blank_current_for_export():
			self.canvas.SetFocus()
			return

		for name in list(self.skipped_images):
			if self._polygon_count(name)==0:
				self._ensure_image_info(name)

		if not self.information:
			wx.MessageBox('No annotations to export.','Error',wx.ICON_ERROR)
			self.canvas.SetFocus()
			return

		self.update_export_only_lock()
		generate_annotation(os.path.dirname(self.image_paths[0]),self.information,self.result_path,self.result_path,self.aug_methods,self.color_map)
		export_only=self.export_only_to_export_folder and not self.same_import_export_folder()
		if not export_only:
			generate_annotation(os.path.dirname(self.image_paths[0]),self.information,os.path.dirname(self.image_paths[0]),self.result_path,[],self.color_map)

		wx.MessageBox('Annotations exported successfully.','Success',wx.ICON_INFORMATION)
		self.refresh_queue_ui()
		self.canvas.SetFocus()



class PanelLv2_AutoAnnotation(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_images=None
		self.path_to_annotator=None
		self.object_kinds=None
		self.detection_threshold={}
		self.filters={}
		self.sliding=False
		self.overlap_ratio=0.2

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the image(s)\nto annotate',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_input,'Select one or more images. Common image formats (jpg, png, tif) are supported. An annotation file will be generated in this folder')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_model=wx.BoxSizer(wx.HORIZONTAL)
		button_model=wx.Button(panel,label='Select a trained Annotator\nfor automatic annotation',size=(300,40))
		button_model.Bind(wx.EVT_BUTTON,self.select_model)
		wx.Button.SetToolTip(button_model,'A trained Annotator can annotate the objects of your interest in images.')
		self.text_model=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_model.Add(button_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_model.Add(self.text_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_filters=wx.BoxSizer(wx.HORIZONTAL)
		button_filters=wx.Button(panel,label='Specify the filters to\nexclude unwanted annotations',size=(300,40))
		button_filters.Bind(wx.EVT_BUTTON,self.specify_filters)
		wx.Button.SetToolTip(button_filters,'Select filters such as area, perimeter, roundness (1 is circle, higer value means less round), height, and width, and specify the minimum and maximum values of these filters.')
		self.text_filters=wx.StaticText(panel,label='Default: None',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_filters.Add(button_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_filters.Add(self.text_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to annotate images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_annotation)
		wx.Button.SetToolTip(button_startannotation,'Automatically annotate objects in images.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_startannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.tif;*.tiff'
		dialog=wx.FileDialog(self,'Select images(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			self.text_input.SetLabel('Select: '+str(len(self.path_to_images))+' images in'+str(path)+'.')
		dialog.Destroy()


	def select_model(self,event):

		annotator_path=os.path.join(the_absolute_current_path,'annotators')
		annotators=[i for i in os.listdir(annotator_path) if os.path.isdir(os.path.join(annotator_path,i))]
		if '__pycache__' in annotators:
			annotators.remove('__pycache__')
		if '__init__' in annotators:
			annotators.remove('__init__')
		if '__init__.py' in annotators:
			annotators.remove('__init__.py')
		annotators.sort()
		if 'Choose a new directory of the Annotator' not in annotators:
			annotators.append('Choose a new directory of the Annotator')

		dialog=wx.SingleChoiceDialog(self,message='Select an Annotator for automatic annotation.',caption='Select an Annotator',choices=annotators)
		if dialog.ShowModal()==wx.ID_OK:
			annotator=dialog.GetStringSelection()
			if annotator=='Choose a new directory of the Annotator':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_annotator=dialog1.GetPath()
				else:
					self.path_to_annotator=None
				dialog1.Destroy()
			else:
				self.path_to_annotator=os.path.join(annotator_path,annotator)
			with open(os.path.join(self.path_to_annotator,'model_parameters.txt')) as f:
				model_parameters=f.read()
			object_names=json.loads(model_parameters)['object_names']
			if len(object_names)>1:
				dialog1=wx.MultiChoiceDialog(self,message='Specify which obejct to annotate',
					caption='Object kind',choices=object_names)
				if dialog1.ShowModal()==wx.ID_OK:
					self.object_kinds=[object_names[i] for i in dialog1.GetSelections()]
				else:
					self.object_kinds=object_names
				dialog1.Destroy()
			else:
				self.object_kinds=object_names
			for object_name in self.object_kinds:
				dialog1=wx.NumberEntryDialog(self,'Detection threshold for '+str(object_name),'Enter an number between 0 and 100','Detection threshold for '+str(object_name),0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					self.detection_threshold[object_name]=int(dialog1.GetValue())/100
				else:
					self.detection_threshold[object_name]=0
				dialog1.Destroy()

			dialog1=wx.MessageDialog(self,'Detect objects with sliding tiles?\n(for images too large to be fit into VRAM)','Tiling the images?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog1.ShowModal()==wx.ID_YES:
				self.sliding=True
				dialog2=wx.NumberEntryDialog(self,'Input the overlapping ratio\nbetween adjacent tiles','A number between 1 and 100:','Overlapping ratio',20,1,100)
				if dialog2.ShowModal()==wx.ID_OK:
					self.overlap_ratio=int(dialog2.GetValue())/100
				else:
					self.overlap_ratio=0.2
				dialog2.Destroy()
				self.text_model.SetLabel('Annotator: '+annotator+'; '+'The object kinds / detection threshold: '+str(self.detection_threshold)+'; Tile overlapping ratio: '+str(self.overlap_ratio)+'.')
			else:
				self.sliding=False
				self.text_model.SetLabel('Annotator: '+annotator+'; '+'The object kinds / detection threshold: '+str(self.detection_threshold)+'; No tiling.')
			dialog1.Destroy()
		dialog.Destroy()

		if self.path_to_annotator is None:
			wx.MessageBox('No Annotator is selected.','No Annotator',wx.ICON_INFORMATION)
			self.text_model.SetLabel('No Annotator is selected.')


	def specify_filters(self,event):

		filters_choices=['area','perimeter','roundness','height','width']

		dialog=wx.MultiChoiceDialog(self,message='Select filters to exclude unwanted annotations',caption='Filters',choices=filters_choices)
		if dialog.ShowModal()==wx.ID_OK:
			selected_filters=[filters_choices[i] for i in dialog.GetSelections()]
		else:
			selected_filters=[]
		dialog.Destroy()

		for ft in selected_filters:
			dialog=wx.NumberEntryDialog(self,'The min value for '+str(ft),'The unit is pixel (except for roundness)','The min value for '+str(ft),0,0,100000000000000)
			values=[0,np.inf]
			if dialog.ShowModal()==wx.ID_OK:
				values[0]=int(dialog.GetValue())
			dialog.Destroy()
			dialog=wx.NumberEntryDialog(self,'The max value (enter 0 for infinity) for '+str(ft),'The unit is pixel (except for roundness)','The max value for '+str(ft),0,0,100000000000000)
			if dialog.ShowModal()==wx.ID_OK:
				value=int(dialog.GetValue())
				if value>0:
					values[1]=value
			dialog.Destroy()
			self.filters[ft]=values

		if len(self.filters)>0:
			self.text_filters.SetLabel('Filters: '+str(self.filters))
		else:
			self.text_filters.SetLabel('NO filters selected.')


	def start_annotation(self,event):

		if self.path_to_images is None or self.path_to_annotator is None:

			wx.MessageBox('No input images(s) / trained Annotator selected.','Error',wx.OK|wx.ICON_ERROR)

		else:
			
			AA=AutoAnnotation(self.path_to_images,self.path_to_annotator,self.object_kinds,detection_threshold=self.detection_threshold,filters=self.filters)
			AA.annotate_images(sliding=self.sliding,overlap=self.overlap_ratio)


