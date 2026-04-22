import os
import cv2
import wx
import json
import random
import threading
import time
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
from .tools import read_annotation,mask_to_polygon,generate_annotation,image_sampler_directory_stats,sample_from_pool



the_absolute_current_path=str(Path(__file__).resolve().parent)



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

		button_bobbysedit=wx.Button(panel,label="Bobby's Interactive Edit",size=(300,40))
		button_bobbysedit.Bind(wx.EVT_BUTTON,self.bobbys_edit)
		wx.Button.SetToolTip(button_bobbysedit,'Open Bobby\'s Interactive Edit mode for review/edit annotation workflow.')
		boxsizer.Add(button_bobbysedit,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_bobbys_sampler=wx.Button(panel,label="Bobby's Image Sampler",size=(300,40))
		button_bobbys_sampler.Bind(wx.EVT_BUTTON,self.bobbys_image_sampler)
		wx.Button.SetToolTip(button_bobbys_sampler,'Sample random images from a pool folder into a dataset folder without overwriting existing files.')
		boxsizer.Add(button_bobbys_sampler,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def manual_annotate(self,event):

		panel=PanelLv2_ManualAnnotation(self.notebook)
		title='Annotate Manually'
		self.notebook.AddPage(panel,title,select=True)


	def auto_annotate(self,event):

		panel=PanelLv2_AutoAnnotation(self.notebook)
		title='Annotate Automatically'
		self.notebook.AddPage(panel,title,select=True)


	def bobbys_edit(self,event):

		from .gui_bobbysedit import PanelLv2_BobbysEdit
		panel=PanelLv2_BobbysEdit(self.notebook)
		title="Bobby's Interactive Edit"
		self.notebook.AddPage(panel,title,select=True)


	def bobbys_image_sampler(self,event):

		panel=PanelLv2_BobbysImageSampler(self.notebook)
		title="Bobby's Image Sampler"
		self.notebook.AddPage(panel,title,select=True)



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



class WindowLv3_AnnotateImages(wx.Frame):

	def __init__(self,parent,title,path_to_images,result_path,color_map,aug_methods,model_cp=None,model_cfg=None):

		monitor=get_monitors()[0]
		monitor_w,monitor_h=monitor.width,monitor.height

		super().__init__(parent,title=title,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-50))

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

		self.init_ui()
		self.load_current_image()


	def sam2_model(self):

		device='cuda' if torch.cuda.is_available() else 'cpu'
		predictor=SAM2ImagePredictor(build_sam2(self.model_cfg,self.model_cp,device=device))
		return predictor


	def init_ui(self):

		panel=wx.Panel(self)
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

		self.scrolled_canvas=wx.ScrolledWindow(panel,style=wx.VSCROLL|wx.HSCROLL)
		self.scrolled_canvas.SetScrollRate(10,10)
		self.canvas=wx.Panel(self.scrolled_canvas,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-50))
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
		self.Bind(wx.EVT_CHAR_HOOK,self.on_key_press)
		self.Show()


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


	def load_current_image(self):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			self.current_image=wx.Image(path,wx.BITMAP_TYPE_ANY)
			img_width,img_height=self.current_image.GetSize()
			self.scrolled_canvas.SetVirtualSize((img_width,img_height))
			self.canvas.SetSize((img_width,img_height))
			self.scrolled_canvas.Scroll(0,0)
			image_name=os.path.basename(path)
			if image_name not in self.information:
				self.information[image_name]={'polygons':[],'class_names':[]}
			self.current_polygon=[]
			self.foreground_points=[]
			self.background_points=[]
			self.scale=1.0
			self.canvas.Refresh()

			if self.AI_help:
				image=Image.open(path)
				image=np.array(image.convert('RGB'))
				self.sam2=self.sam2_model()
				self.sam2.set_image(image)


	def previous_image(self,event):

		if self.image_paths and self.current_image_id>0:
			self.current_image_id-=1
			self.load_current_image()
		self.canvas.SetFocus()


	def next_image(self,event):

		if self.image_paths and self.current_image_id<len(self.image_paths)-1:
			self.current_image_id+=1
			self.load_current_image()
		self.canvas.SetFocus()


	def delete_image(self,event):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			self.image_paths.remove(path)
			image_name=os.path.basename(path)
			del self.information[image_name]
			self.load_current_image()
		self.canvas.SetFocus()


	def on_paint(self,event):

		if self.current_image is None:
			return

		dc=wx.PaintDC(self.canvas)
		w,h=self.current_image.GetSize()
		scaled_image=self.current_image.Scale(int(w*self.scale),int(h*self.scale),wx.IMAGE_QUALITY_HIGH)
		dc.DrawBitmap(wx.Bitmap(scaled_image),0,0,True)
		image_name=os.path.basename(self.image_paths[self.current_image_id])
		polygons=self.information[image_name]['polygons']
		class_names=self.information[image_name]['class_names']

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
			for i,polygon in enumerate(self.information[image_name]['polygons']):
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
				polygons=self.information[image_name]['polygons']
				class_names=self.information[image_name]['class_names']
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

		self.canvas.Refresh()


	def on_key_press(self,event):

		key_code=event.GetKeyCode()

		if event.GetKeyCode()==wx.WXK_RETURN:
			if len(self.current_polygon)>2:
				classnames=sorted(list(self.color_map.keys()))
				current_index=classnames.index(self.current_classname)
				dialog=wx.SingleChoiceDialog(self,message='Choose object class name',caption='Class Name',choices=classnames)
				dialog.SetSelection(current_index)
				if dialog.ShowModal()==wx.ID_OK:
					self.current_classname=dialog.GetStringSelection()
					if len(self.current_polygon)>0:
						self.current_polygon.append(self.current_polygon[0])
						image_name=os.path.basename(self.image_paths[self.current_image_id])
						self.information[image_name]['polygons'].append(self.current_polygon)
						self.information[image_name]['class_names'].append(self.current_classname)
				dialog.Destroy()
				self.current_polygon=[]
				self.foreground_points=[]
				self.background_points=[]
				self.canvas.Refresh()
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


	def export_annotations(self,event):

		if not self.information:
			wx.MessageBox('No annotations to export.','Error',wx.ICON_ERROR)
			return

		generate_annotation(os.path.dirname(self.image_paths[0]),self.information,self.result_path,self.result_path,self.aug_methods,self.color_map)
		generate_annotation(os.path.dirname(self.image_paths[0]),self.information,os.path.dirname(self.image_paths[0]),self.result_path,[],self.color_map)

		wx.MessageBox('Annotations exported successfully.','Success',wx.ICON_INFORMATION)

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
					self.overlap_ratio=int(dialog1.GetValue())/100
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


class PanelLv2_BobbysImageSampler(wx.Panel):

	_IMG_SAMPLER_BTN_SIZE=(300,40)

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.image_pool_dir=None
		self.dataset_dir=None
		self.n_samples=100
		self.seed=42
		self._sampler_elapse_timer=None
		self._sampler_running=False
		self._sampler_start_time=0.0
		self._sampler_progress_cur=0
		self._sampler_progress_tot=0

		self.display_window()


	@staticmethod
	def _fmt_hms(seconds):

		sec=max(0,int(round(seconds)))
		h,rem=divmod(sec,3600)
		m,s=divmod(rem,60)
		return '{}:{:02d}:{:02d}'.format(h,m,s)


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		module_pool=wx.BoxSizer(wx.HORIZONTAL)
		button_pool=wx.Button(panel,label='Select the image\npool folder',size=self._IMG_SAMPLER_BTN_SIZE)
		button_pool.Bind(wx.EVT_BUTTON,self.select_pool_folder)
		self.text_pool=wx.StaticText(panel,label='Pool folder: (not selected)',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_pool.Add(button_pool,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_pool.Add(self.text_pool,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_pool,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_dataset=wx.BoxSizer(wx.HORIZONTAL)
		button_dataset=wx.Button(panel,label='Select the\ndataset folder',size=self._IMG_SAMPLER_BTN_SIZE)
		button_dataset.Bind(wx.EVT_BUTTON,self.select_dataset_folder)
		self.text_dataset=wx.StaticText(panel,label='Dataset folder: (not selected)',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_dataset.Add(button_dataset,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_dataset.Add(self.text_dataset,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_dataset,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_eligible=wx.BoxSizer(wx.HORIZONTAL)
		elig_ph=wx.Panel(panel,size=self._IMG_SAMPLER_BTN_SIZE)
		self.text_eligible=wx.StaticText(panel,label='Eligible images: -',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		self.text_eligible_warn=wx.StaticText(panel,label='',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		self.text_eligible_warn.SetForegroundColour(wx.Colour(255,140,0))
		elig_col=wx.BoxSizer(wx.VERTICAL)
		elig_col.Add(self.text_eligible,0,wx.EXPAND)
		elig_col.Add(self.text_eligible_warn,0,wx.EXPAND)
		module_eligible.Add(elig_ph,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_eligible.Add(elig_col,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_eligible,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_n=wx.BoxSizer(wx.HORIZONTAL)
		button_n=wx.Button(panel,label='Set number of images\nto sample',size=self._IMG_SAMPLER_BTN_SIZE)
		button_n.Bind(wx.EVT_BUTTON,self.set_n_samples)
		self.text_n=wx.StaticText(panel,label='N: 100',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_n.Add(button_n,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_n.Add(self.text_n,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_n,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_seed=wx.BoxSizer(wx.HORIZONTAL)
		button_seed=wx.Button(panel,label='Set random\nseed',size=self._IMG_SAMPLER_BTN_SIZE)
		button_seed.Bind(wx.EVT_BUTTON,self.set_seed)
		self.text_seed=wx.StaticText(panel,label='Seed: 42',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_seed.Add(button_seed,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_seed.Add(self.text_seed,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_seed,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		self._sampler_step_buttons=(button_pool,button_dataset,button_n,button_seed)

		self.button_run=wx.Button(panel,label="Run Bobby's Image Sampler",size=self._IMG_SAMPLER_BTN_SIZE)
		self.button_run.Bind(wx.EVT_BUTTON,self.run_sampler)
		boxsizer.Add(0,5,0)
		boxsizer.Add(self.button_run,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		self.progress_holder=wx.Panel(panel)
		prog_sz=wx.BoxSizer(wx.VERTICAL)
		self.gauge=wx.Gauge(self.progress_holder,range=1,size=(-1,18),style=wx.GA_HORIZONTAL)
		prog_sz.Add(self.gauge,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP,border=10)
		self.text_copying=wx.StaticText(self.progress_holder,label='',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		prog_sz.Add(self.text_copying,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP,border=5)
		self.text_elapsed_p=wx.StaticText(self.progress_holder,label='Elapsed: 0:00:00',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		self.text_eta_p=wx.StaticText(self.progress_holder,label='Estimated completion: Calculating...',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		prog_sz.Add(self.text_elapsed_p,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP,border=5)
		prog_sz.Add(self.text_eta_p,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP|wx.BOTTOM,border=5)
		self.progress_holder.SetSizer(prog_sz)
		self.progress_holder.Hide()
		boxsizer.Add(self.progress_holder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)
		self.Bind(wx.EVT_WINDOW_DESTROY,self._sampler_on_destroy)
		self.Centre()
		self.Show(True)
		self.refresh_descriptions()


	def _sampler_on_destroy(self,event):

		self._sampler_running=False
		self._sampler_stop_elapse_timer()
		event.Skip()


	def _sampler_stop_elapse_timer(self):

		t=self._sampler_elapse_timer
		if t is not None:
			t.Stop()
			self._sampler_elapse_timer=None


	def refresh_descriptions(self):

		pool_dir=self.image_pool_dir or ''
		ds_dir=self.dataset_dir or ''
		stats=image_sampler_directory_stats(pool_dir,ds_dir)
		pool_c=stats['pool_count']
		ds_img=stats['dataset_image_count']
		elig=stats['eligible_count']

		if self.image_pool_dir:
			self.text_pool.SetLabel('Pool folder: {}\n{} images found in pool'.format(self.image_pool_dir,pool_c))
		else:
			self.text_pool.SetLabel('Pool folder: (not selected)')

		if self.dataset_dir:
			self.text_dataset.SetLabel('Dataset folder: {}\n{} images already in dataset'.format(self.dataset_dir,ds_img))
		else:
			self.text_dataset.SetLabel('Dataset folder: (not selected)')

		self.text_eligible.SetLabel('Eligible images: {} images eligible for sampling'.format(elig))

		if self.n_samples>elig:
			self.text_eligible_warn.SetLabel('Only {} images available — all will be copied.'.format(elig))
			self.text_eligible_warn.Show()
		else:
			self.text_eligible_warn.SetLabel('')
			self.text_eligible_warn.Hide()

		self.text_n.SetLabel('N: {}'.format(self.n_samples))
		self.text_seed.SetLabel('Seed: {}'.format(self.seed))
		self.Layout()


	def select_pool_folder(self,event):

		if self._sampler_running:
			return
		dialog=wx.DirDialog(self,'Select the image pool folder','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.image_pool_dir=dialog.GetPath()
		dialog.Destroy()
		self.refresh_descriptions()


	def select_dataset_folder(self,event):

		if self._sampler_running:
			return
		dialog=wx.DirDialog(self,'Select the dataset folder','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.dataset_dir=dialog.GetPath()
		dialog.Destroy()
		self.refresh_descriptions()


	def set_n_samples(self,event):

		if self._sampler_running:
			return
		dialog=wx.NumberEntryDialog(self,'How many images to copy from the pool into the dataset.','Integer:','Number of images to sample',self.n_samples,0,1000000000)
		if dialog.ShowModal()==wx.ID_OK:
			self.n_samples=int(dialog.GetValue())
		dialog.Destroy()
		self.refresh_descriptions()


	def set_seed(self,event):

		if self._sampler_running:
			return
		dialog=wx.NumberEntryDialog(self,'Random seed for sampling reproducibility.','Integer:','Random seed',self.seed,-2147483648,2147483647)
		if dialog.ShowModal()==wx.ID_OK:
			self.seed=int(dialog.GetValue())
		dialog.Destroy()
		self.refresh_descriptions()


	def _sampler_refresh_elapse_eta(self):

		elapsed=time.time()-self._sampler_start_time
		self.text_elapsed_p.SetLabel('Elapsed: '+self._fmt_hms(elapsed))
		tot=self._sampler_progress_tot
		cur=self._sampler_progress_cur
		if elapsed<3.0:
			eta_lbl='Estimated completion: Calculating...'
		elif tot<=0:
			eta_lbl='Estimated completion: Calculating...'
		elif cur<=0:
			eta_lbl='Estimated completion: Estimating...'
		else:
			rate=elapsed/float(cur)
			rem=max(0.0,(tot-cur)*rate)
			eta_lbl='Estimated completion: '+self._fmt_hms(rem)
		self.text_eta_p.SetLabel(eta_lbl)


	def _sampler_elapse_tick(self):

		if not self._sampler_running:
			return
		self._sampler_refresh_elapse_eta()
		self._sampler_elapse_timer=wx.CallLater(1000,self._sampler_elapse_tick)


	def _sampler_on_progress(self,current,total,basename):

		self._sampler_progress_cur=current
		self._sampler_progress_tot=total
		rg=max(1,total)
		self.gauge.SetRange(rg)
		self.gauge.SetValue(min(current,rg))
		if total<=0:
			self.text_copying.SetLabel('')
		elif current<=0:
			self.text_copying.SetLabel('Preparing...')
		else:
			self.text_copying.SetLabel('Copying image {} of {}...'.format(current,total))
		self._sampler_refresh_elapse_eta()
		self.Layout()


	def _sampler_worker(self):

		def cb(cur,tot,base):
			wx.CallAfter(self._sampler_on_progress,cur,tot,base)

		try:
			result=sample_from_pool(self.image_pool_dir,self.dataset_dir,self.n_samples,self.seed,progress_callback=cb)
			wx.CallAfter(self._sampler_on_finished,result,None)
		except Exception as e:
			wx.CallAfter(self._sampler_on_finished,None,e)


	def _sampler_set_controls_busy(self,busy):

		for b in self._sampler_step_buttons:
			b.Enable(not busy)
		self.button_run.Enable(not busy)


	def _sampler_on_finished(self,result,error):

		self._sampler_running=False
		self._sampler_stop_elapse_timer()
		elapsed=time.time()-self._sampler_start_time
		self.text_elapsed_p.SetLabel('Elapsed: '+self._fmt_hms(elapsed))
		self.text_eta_p.SetLabel('Estimated completion: Done')
		tot=self._sampler_progress_tot
		rg=max(1,tot)
		self.gauge.SetRange(rg)
		self.gauge.SetValue(rg)
		if tot>0:
			self.text_copying.SetLabel('Copying image {} of {}...'.format(tot,tot))
		else:
			self.text_copying.SetLabel('')

		if error is not None:
			wx.MessageBox(str(error),"Bobby's Image Sampler",wx.OK|wx.ICON_ERROR)
			self.progress_holder.Hide()
			self._sampler_set_controls_busy(False)
			self.Layout()
			return

		parts=[
			'Copied: {}'.format(result['copied']),
			'Requested: {}'.format(result['requested']),
			'Eligible available: {}'.format(result['available']),
		]
		if result.get('warning'):
			parts.append('')
			parts.append(result['warning'])
		msg='\n'.join(parts)
		dlg=wx.MessageDialog(self,msg,"Bobby's Image Sampler",wx.OK|wx.ICON_INFORMATION)
		dlg.ShowModal()
		dlg.Destroy()

		self.progress_holder.Hide()
		self.gauge.SetValue(0)
		self.gauge.SetRange(1)
		self.text_copying.SetLabel('')
		self.text_elapsed_p.SetLabel('Elapsed: 0:00:00')
		self.text_eta_p.SetLabel('Estimated completion: Calculating...')

		self.image_pool_dir=None
		self.dataset_dir=None
		self.n_samples=100
		self.seed=42
		self._sampler_set_controls_busy(False)
		self.refresh_descriptions()


	def run_sampler(self,event):

		if self._sampler_running:
			return
		if not self.image_pool_dir or not self.dataset_dir:
			wx.MessageBox('Please select both the image pool folder and the dataset folder.',"Bobby's Image Sampler",wx.OK|wx.ICON_WARNING)
			return

		self._sampler_set_controls_busy(True)
		self.progress_holder.Show()
		self._sampler_running=True
		self._sampler_start_time=time.time()
		self._sampler_progress_cur=0
		self._sampler_progress_tot=0
		self.gauge.SetRange(1)
		self.gauge.SetValue(0)
		self.text_copying.SetLabel('Starting...')
		self.text_elapsed_p.SetLabel('Elapsed: 0:00:00')
		self.text_eta_p.SetLabel('Estimated completion: Calculating...')
		self._sampler_stop_elapse_timer()
		self._sampler_refresh_elapse_eta()
		self._sampler_elapse_timer=wx.CallLater(1000,self._sampler_elapse_tick)
		self.Layout()

		threading.Thread(target=self._sampler_worker,daemon=True).start()

