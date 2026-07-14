"""GUI for merging annotated image folders into one dataset."""

import json
import os
import random

import wx
from screeninfo import get_monitors

from .dataset_merge import (
	COLLISION_KEEP_A,
	COLLISION_KEEP_B,
	COLLISION_REVIEW,
	MergeError,
	MergeOptions,
	format_preview,
	merge_datasets,
	policy_description,
	policy_label,
	preview_merge,
	summarize_folder,
)
from .tools import read_annotation


class _AnnotationPreviewCanvas(wx.Panel):
	"""Read-only image + polygon overlay (same paint style as manual annotate)."""

	def __init__(self,parent):

		super().__init__(parent)
		self.image=None
		self.polygons=[]
		self.class_names=[]
		self.color_map={}
		self.scale=1.0
		self.min_scale=0.1
		self.SetBackgroundColour('black')
		self.Bind(wx.EVT_PAINT,self.on_paint)
		self.Bind(wx.EVT_SIZE,self.on_size)


	def set_content(self,image_path,polygons,class_names,color_map):

		self.color_map=color_map or {}
		self.polygons=polygons or []
		self.class_names=class_names or []
		if image_path and os.path.isfile(image_path):
			self.image=wx.Image(image_path,wx.BITMAP_TYPE_ANY)
		else:
			self.image=None
		self.fit_to_view()
		self.Refresh()


	def fit_to_view(self):

		if self.image is None:
			return
		img_w,img_h=self.image.GetSize()
		cw,ch=self.GetClientSize()
		if cw<=0 or ch<=0 or img_w<=0 or img_h<=0:
			self.scale=1.0
			return
		self.scale=max(min(cw/img_w,ch/img_h),self.min_scale)


	def on_size(self,event):

		self.fit_to_view()
		self.Refresh()
		event.Skip()


	def on_paint(self,event):

		dc=wx.PaintDC(self)
		dc.Clear()
		if self.image is None:
			dc.SetTextForeground(wx.Colour(200,200,200))
			dc.DrawText('No image',10,10)
			return
		w,h=self.image.GetSize()
		scaled=self.image.Scale(max(1,int(w*self.scale)),max(1,int(h*self.scale)),wx.IMAGE_QUALITY_HIGH)
		dc.DrawBitmap(wx.Bitmap(scaled),0,0,True)
		for i,polygon in enumerate(self.polygons):
			classname=self.class_names[i] if i<len(self.class_names) else None
			color=self.color_map.get(classname,(0,255,0)) if classname else (0,255,0)
			pen=wx.Pen(wx.Colour(*color),width=2)
			dc.SetPen(pen)
			dc.SetBrush(wx.Brush(wx.Colour(*color),style=wx.BRUSHSTYLE_TRANSPARENT))
			points=[(int(x*self.scale),int(y*self.scale)) for x,y in polygon]
			if len(points)>=2:
				dc.DrawLines(points+([points[0]] if points[0]!=points[-1] else []))



class WindowLv3_CollisionReview(wx.Dialog):
	"""Side-by-side review of colliding filenames: left = Dataset A, right = Dataset B."""

	def __init__(self,parent,folder_a,folder_b,collision_names,color_map):

		monitor=get_monitors()[0]
		super().__init__(
			parent,
			title='Review Name Collisions',
			style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER|wx.MAXIMIZE_BOX,
			size=(monitor.width-40,monitor.height-100),
		)
		self.folder_a=folder_a
		self.folder_b=folder_b
		self.collision_names=list(collision_names)
		self.color_map=color_map
		self.info_a=read_annotation(folder_a,color_map=color_map)
		self.info_b=read_annotation(folder_b,color_map=color_map)
		self.choices={}
		self.current_index=0

		self.init_ui()
		self.load_current()
		self.CentreOnParent()


	def init_ui(self):

		root=wx.BoxSizer(wx.VERTICAL)

		toolbar=wx.BoxSizer(wx.HORIZONTAL)
		self.prev_button=wx.Button(self,label='← Prev',size=(120,30))
		self.prev_button.Bind(wx.EVT_BUTTON,self.previous_collision)
		toolbar.Add(self.prev_button,0,wx.ALL,4)

		self.next_button=wx.Button(self,label='Next →',size=(120,30))
		self.next_button.Bind(wx.EVT_BUTTON,self.next_collision)
		toolbar.Add(self.next_button,0,wx.ALL,4)

		self.text_status=wx.StaticText(self,label='',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_MIDDLE)
		toolbar.Add(self.text_status,1,wx.ALIGN_CENTER_VERTICAL|wx.LEFT|wx.RIGHT,10)
		root.Add(toolbar,0,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP,6)

		self.text_filename=wx.StaticText(self,label='',style=wx.ALIGN_CENTER)
		root.Add(self.text_filename,0,wx.EXPAND|wx.ALL,4)

		panes=wx.BoxSizer(wx.HORIZONTAL)

		left=wx.BoxSizer(wx.VERTICAL)
		left.Add(wx.StaticText(self,label='Dataset A (left)'),0,wx.ALIGN_CENTER|wx.BOTTOM,4)
		self.canvas_a=_AnnotationPreviewCanvas(self)
		left.Add(self.canvas_a,1,wx.EXPAND|wx.ALL,4)
		self.text_a_meta=wx.StaticText(self,label='',style=wx.ALIGN_CENTER)
		left.Add(self.text_a_meta,0,wx.EXPAND|wx.BOTTOM,4)
		self.keep_a_button=wx.Button(self,label='Keep Left (A)',size=(200,36))
		self.keep_a_button.Bind(wx.EVT_BUTTON,self.choose_a)
		left.Add(self.keep_a_button,0,wx.ALIGN_CENTER|wx.BOTTOM,8)
		panes.Add(left,1,wx.EXPAND)

		right=wx.BoxSizer(wx.VERTICAL)
		right.Add(wx.StaticText(self,label='Dataset B (right)'),0,wx.ALIGN_CENTER|wx.BOTTOM,4)
		self.canvas_b=_AnnotationPreviewCanvas(self)
		right.Add(self.canvas_b,1,wx.EXPAND|wx.ALL,4)
		self.text_b_meta=wx.StaticText(self,label='',style=wx.ALIGN_CENTER)
		right.Add(self.text_b_meta,0,wx.EXPAND|wx.BOTTOM,4)
		self.keep_b_button=wx.Button(self,label='Keep Right (B)',size=(200,36))
		self.keep_b_button.Bind(wx.EVT_BUTTON,self.choose_b)
		right.Add(self.keep_b_button,0,wx.ALIGN_CENTER|wx.BOTTOM,8)
		panes.Add(right,1,wx.EXPAND)

		root.Add(panes,1,wx.EXPAND|wx.LEFT|wx.RIGHT,6)

		footer=wx.BoxSizer(wx.HORIZONTAL)
		self.text_progress=wx.StaticText(self,label='')
		footer.Add(self.text_progress,1,wx.ALIGN_CENTER_VERTICAL|wx.LEFT,8)
		cancel=wx.Button(self,wx.ID_CANCEL,label='Cancel')
		footer.Add(cancel,0,wx.ALL,6)
		self.done_button=wx.Button(self,wx.ID_OK,label='Done - use these choices')
		self.done_button.Bind(wx.EVT_BUTTON,self.on_done)
		footer.Add(self.done_button,0,wx.ALL,6)
		root.Add(footer,0,wx.EXPAND|wx.BOTTOM,4)

		self.SetSizer(root)
		self.Bind(wx.EVT_CHAR_HOOK,self.on_key_press)


	def _side_info(self,info,name):

		entry=info.get(name,{'polygons':[],'class_names':[]})
		return entry.get('polygons',[]),entry.get('class_names',[])


	def load_current(self):

		if not self.collision_names:
			return
		name=self.collision_names[self.current_index]
		path_a=os.path.join(self.folder_a,name)
		path_b=os.path.join(self.folder_b,name)
		polys_a,classes_a=self._side_info(self.info_a,name)
		polys_b,classes_b=self._side_info(self.info_b,name)
		self.canvas_a.set_content(path_a,polys_a,classes_a,self.color_map)
		self.canvas_b.set_content(path_b,polys_b,classes_b,self.color_map)
		self.text_filename.SetLabel(f'Filename: {name} ({self.current_index+1} / {len(self.collision_names)})')
		self.text_a_meta.SetLabel(f'{len(polys_a)} annotation(s)')
		self.text_b_meta.SetLabel(f'{len(polys_b)} annotation(s)')
		choice=self.choices.get(name)
		if choice=='a':
			choice_label='Current choice: Keep Left (A)'
		elif choice=='b':
			choice_label='Current choice: Keep Right (B)'
		else:
			choice_label='Current choice: not decided'
		self.text_status.SetLabel(choice_label)
		decided=sum(1 for n in self.collision_names if n in self.choices)
		self.text_progress.SetLabel(f'Decided {decided} / {len(self.collision_names)}')
		self.done_button.Enable(decided==len(self.collision_names))


	def previous_collision(self,event):

		if self.current_index>0:
			self.current_index-=1
			self.load_current()


	def next_collision(self,event):

		if self.current_index<len(self.collision_names)-1:
			self.current_index+=1
			self.load_current()


	def choose_a(self,event):

		name=self.collision_names[self.current_index]
		self.choices[name]='a'
		if self.current_index<len(self.collision_names)-1:
			self.current_index+=1
		self.load_current()


	def choose_b(self,event):

		name=self.collision_names[self.current_index]
		self.choices[name]='b'
		if self.current_index<len(self.collision_names)-1:
			self.current_index+=1
		self.load_current()


	def on_key_press(self,event):

		key=event.GetKeyCode()
		if key in (wx.WXK_LEFT,ord('a'),ord('A')):
			self.choose_a(event)
		elif key in (wx.WXK_RIGHT,ord('b'),ord('B')):
			self.choose_b(event)
		elif key==wx.WXK_UP:
			self.previous_collision(event)
		elif key==wx.WXK_DOWN:
			self.next_collision(event)
		else:
			event.Skip()


	def on_done(self,event):

		if len(self.choices)!=len(self.collision_names):
			wx.MessageBox('Choose A or B for every colliding filename before continuing.','Incomplete',wx.OK|wx.ICON_WARNING)
			return
		self.EndModal(wx.ID_OK)


	def get_choices(self):

		return dict(self.choices)



class MergePreviewDialog(wx.Dialog):
	"""Popup with merge summary text and a Refresh control."""

	def __init__(self,parent,owner):

		super().__init__(
			parent,
			title='Merge preview',
			style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER,
			size=(520,420),
		)
		self.owner=owner

		root=wx.BoxSizer(wx.VERTICAL)

		header=wx.BoxSizer(wx.HORIZONTAL)
		caption=wx.StaticText(
			self,
			label='What will be combined from Dataset A and Dataset B',
			style=wx.ALIGN_LEFT,
		)
		caption_font=caption.GetFont()
		if caption_font.IsOk():
			caption_font=wx.Font(caption_font)
			caption_font.MakeBold()
			caption.SetFont(caption_font)
		header.Add(caption,1,wx.ALIGN_CENTER_VERTICAL|wx.RIGHT,8)

		self.button_refresh=wx.Button(self,label='Refresh',size=(70,28))
		self.button_refresh.Bind(wx.EVT_BUTTON,self.on_refresh)
		wx.Button.SetToolTip(self.button_refresh,'Refresh merge preview')
		header.Add(self.button_refresh,0,wx.ALIGN_CENTER_VERTICAL)
		root.Add(header,0,wx.EXPAND|wx.ALL,12)

		self.text_preview=wx.TextCtrl(
			self,
			value='',
			style=wx.TE_MULTILINE|wx.TE_READONLY|wx.TE_WORDWRAP,
		)
		root.Add(self.text_preview,1,wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM,12)

		close=wx.Button(self,wx.ID_CLOSE,label='Close')
		close.Bind(wx.EVT_BUTTON,lambda event: self.EndModal(wx.ID_CLOSE))
		root.Add(close,0,wx.ALIGN_RIGHT|wx.RIGHT|wx.BOTTOM,12)

		self.SetSizer(root)
		self.CentreOnParent()
		self.refresh_content()


	def on_refresh(self,event):

		self.refresh_content()


	def refresh_content(self):

		self.text_preview.SetValue(self.owner.build_preview_text())
		self.text_preview.ShowPosition(0)



class PanelLv2_MergeDatasets(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.folder_a=None
		self.folder_b=None
		self.folder_out=None
		self.collision_policy=COLLISION_REVIEW
		self.preview=None

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_a=wx.BoxSizer(wx.HORIZONTAL)
		button_a=wx.Button(panel,label='Select the Dataset A folder\nto merge',size=(300,40))
		button_a.Bind(wx.EVT_BUTTON,self.select_folder_a)
		wx.Button.SetToolTip(button_a,'First source folder of images and optional annotations.json. Original folders are not modified.')
		self.text_a=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_a.Add(button_a,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_a.Add(self.text_a,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_a,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_b=wx.BoxSizer(wx.HORIZONTAL)
		button_b=wx.Button(panel,label='Select the Dataset B folder\nto merge',size=(300,40))
		button_b.Bind(wx.EVT_BUTTON,self.select_folder_b)
		wx.Button.SetToolTip(button_b,'Second source folder of images and optional annotations.json. Original folders are not modified.')
		self.text_b=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_b.Add(button_b,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_b.Add(self.text_b,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_b,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_out=wx.BoxSizer(wx.HORIZONTAL)
		button_out=wx.Button(panel,label='Select a folder to store\nthe merged dataset',size=(300,40))
		button_out.Bind(wx.EVT_BUTTON,self.select_folder_out)
		wx.Button.SetToolTip(button_out,'Combined destination folder. Must differ from Dataset A and B. Images are copied here and a merged annotations.json is written.')
		self.text_out=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_out.Add(button_out,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_out.Add(self.text_out,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_out,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_policy=wx.BoxSizer(wx.HORIZONTAL)
		button_policy=wx.Button(panel,label='Specify the policy for\nname collisions',size=(300,40))
		button_policy.Bind(wx.EVT_BUTTON,self.choose_collision_policy)
		wx.Button.SetToolTip(
			button_policy,
			'Only applies when the same file name exists in both folders.\n'
			'• Review each collision (default): side-by-side window; choose Left (A) or Right (B) per file.\n'
			'• Keep Dataset A for all: always use A’s image and annotations.\n'
			'• Keep Dataset B for all: always use B’s image and annotations.',
		)
		self.text_policy=wx.StaticText(panel,label='Default: Review each collision',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_policy.Add(button_policy,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_policy.Add(self.text_policy,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_policy,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_preview=wx.BoxSizer(wx.HORIZONTAL)
		button_preview=wx.Button(panel,label='View Merge Preview',size=(300,40))
		button_preview.Bind(wx.EVT_BUTTON,self.open_merge_preview)
		wx.Button.SetToolTip(button_preview,'Open a summary of images, annotations, collisions, and the output folder.')
		self.text_preview_status=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_preview.Add(button_preview,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_preview.Add(self.text_preview_status,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_preview,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_start=wx.Button(panel,label='Start to merge datasets',size=(300,40))
		button_start.Bind(wx.EVT_BUTTON,self.start_merge)
		wx.Button.SetToolTip(button_start,'Copy images into the output folder and write a single merged annotations.json.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_start,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_folder_a(self,event):

		dialog=wx.DirDialog(self,'Select Dataset A','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.folder_a=dialog.GetPath()
			try:
				summary=summarize_folder(self.folder_a)
				self.text_a.SetLabel('The Dataset A folder: '+self.folder_a+' ('+summary+').')
			except MergeError as exc:
				self.folder_a=None
				wx.MessageBox(str(exc),'Error',wx.OK|wx.ICON_ERROR)
		dialog.Destroy()
		self.refresh_preview()


	def select_folder_b(self,event):

		dialog=wx.DirDialog(self,'Select Dataset B','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.folder_b=dialog.GetPath()
			try:
				summary=summarize_folder(self.folder_b)
				self.text_b.SetLabel('The Dataset B folder: '+self.folder_b+' ('+summary+').')
			except MergeError as exc:
				self.folder_b=None
				wx.MessageBox(str(exc),'Error',wx.OK|wx.ICON_ERROR)
		dialog.Destroy()
		self.refresh_preview()


	def select_folder_out(self,event):

		dialog=wx.DirDialog(self,'Select output folder','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.folder_out=dialog.GetPath()
			self.text_out.SetLabel('The merged dataset will be in: '+self.folder_out+'.')
		dialog.Destroy()
		self.refresh_preview()


	def choose_collision_policy(self,event):

		choices=[
			'Review each collision (choose Left A or Right B)',
			'Keep Dataset A for all collisions',
			'Keep Dataset B for all collisions',
		]
		policy_order=[COLLISION_REVIEW,COLLISION_KEEP_A,COLLISION_KEEP_B]
		dialog=wx.SingleChoiceDialog(
			self,
			'When the same file name appears in Dataset A and Dataset B,\nchoose how to resolve the collision:',
			'Name collision policy',
			choices,
		)
		dialog.SetSelection(policy_order.index(self.collision_policy) if self.collision_policy in policy_order else 0)
		if dialog.ShowModal()==wx.ID_OK:
			selection=dialog.GetSelection()
			self.collision_policy=policy_order[selection]
			self.text_policy.SetLabel('The name collision policy: '+policy_description(self.collision_policy)+'.')
		dialog.Destroy()


	def _prompt_collision_outline_colors(self):
		"""Ask for outline colors per class using the same ColorPicker as manual annotate."""

		from .gui_annotating import ColorPicker

		classnames=[]
		for folder in (self.folder_a,self.folder_b):
			json_path=os.path.join(folder,'annotations.json')
			if not os.path.isfile(json_path):
				continue
			annotation=json.load(open(json_path))
			for i in annotation.get('categories',[]):
				if i.get('id',0)>0 and i['name'] not in classnames:
					classnames.append(i['name'])

		color_map={}
		if not classnames:
			return color_map

		for classname in classnames:
			seed_hex='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
			dialog=ColorPicker(self,str(classname),[classname,seed_hex])
			if dialog.ShowModal()==wx.ID_OK:
				(r,b,g,_)=dialog.color_picker.GetColour()
				color_map[classname]=(r,b,g)
			else:
				color_map[classname]=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
			dialog.Destroy()
		return color_map


	def open_merge_preview(self,event):

		dialog=MergePreviewDialog(self,self)
		dialog.ShowModal()
		dialog.Destroy()
		self.update_preview_status()


	def build_preview_text(self):

		if self.folder_a is None or self.folder_b is None:
			self.preview=None
			msg='Select Dataset A and Dataset B to see a merge preview.'
			if self.folder_out:
				msg+='\n\nOutput folder: '+self.folder_out
			else:
				msg+='\n\nOutput folder: not selected yet'
			return msg
		try:
			self.preview=preview_merge(self.folder_a,self.folder_b)
			return format_preview(self.preview,folder_out=self.folder_out)
		except MergeError as exc:
			self.preview=None
			return str(exc)


	def update_preview_status(self):

		if self.preview is None:
			if self.folder_a is None or self.folder_b is None:
				self.text_preview_status.SetLabel('Select Dataset A and B, then use View Merge Preview.')
			else:
				self.text_preview_status.SetLabel('Use View Merge Preview to see the summary.')
			return
		n=self.preview.unique_image_count
		c=len(self.preview.both)
		out=self.folder_out if self.folder_out else 'not selected'
		self.text_preview_status.SetLabel(
			'Last preview: '+str(n)+' unique images, '+str(c)+' collisions; output: '+out+'.'
		)


	def refresh_preview(self):

		self.build_preview_text()
		self.update_preview_status()


	def start_merge(self,event):

		if self.folder_a is None or self.folder_b is None or self.folder_out is None:
			wx.MessageBox('Select Dataset A, Dataset B, and an output folder.','Error',wx.OK|wx.ICON_ERROR)
			return

		try:
			if os.path.realpath(self.folder_a)==os.path.realpath(self.folder_b):
				wx.MessageBox('Select two different folders.','Error',wx.OK|wx.ICON_ERROR)
				return
			if os.path.realpath(self.folder_out) in (os.path.realpath(self.folder_a),os.path.realpath(self.folder_b)):
				wx.MessageBox('Output must be a separate folder.','Error',wx.OK|wx.ICON_ERROR)
				return
		except OSError:
			pass

		if self.preview is None:
			self.refresh_preview()
		if self.preview is None:
			wx.MessageBox('Could not build merge preview.','Error',wx.OK|wx.ICON_ERROR)
			return

		existing=[n for n in os.listdir(self.folder_out) if not n.startswith('.')]
		if existing:
			dialog=wx.MessageDialog(
				self,
				'Output folder is not empty ('+str(len(existing))+' files/folders). Continue?\n\n'
				'Images will be copied into this folder and annotations.json will be written '
				'(existing annotations.json is backed up as annotations.json.bak).',
				'Output folder not empty',
				wx.YES_NO|wx.ICON_WARNING,
			)
			if dialog.ShowModal()!=wx.ID_YES:
				dialog.Destroy()
				return
			dialog.Destroy()

		n_collisions=len(self.preview.both)
		policy_line=policy_description(self.collision_policy)
		if n_collisions:
			collision_block=(
				'COLLISION POLICY\n'
				'  '+policy_line+'\n'
				'  Name collisions to resolve: '+str(n_collisions)+'\n\n'
			)
		else:
			collision_block=(
				'COLLISION POLICY\n'
				'  '+policy_line+'\n'
				'  No name collisions (policy will not change any files).\n\n'
			)

		confirm=wx.MessageDialog(
			self,
			'Backup recommended: copy important folders before merging.\n\n'
			+collision_block
			+'EZannot will:\n'
			'  • Copy images to\n    '+self.folder_out+'\n'
			'  • Write merged annotations.json\n'
			'  • Save annotations.json.bak if output already has annotations.json\n\n'
			'Original Dataset A and B folders are not modified.\n\n'
			'Proceed with merge?',
			'Confirm merge',
			wx.YES_NO|wx.ICON_INFORMATION,
		)
		if confirm.ShowModal()!=wx.ID_YES:
			confirm.Destroy()
			return
		confirm.Destroy()

		collision_choices=None
		if self.collision_policy==COLLISION_REVIEW and self.preview.both:
			review_intro=wx.MessageDialog(
				self,
				str(n_collisions)+' image name(s) appear in both Dataset A and Dataset B.\n\n'
				'What this means:\n'
				'  For each colliding filename, only one version can go into the\n'
				'  merged folder. You will choose Dataset A (left) or Dataset B (right).\n\n'
				'What happens next:\n'
				'  1. Pick outline colors for each object class.\n'
				'  2. A review window opens: A on the left, B on the right,\n'
				'     with annotations drawn on both images.\n'
				'  3. Press Keep Left (A) or Keep Right (B) for each file\n'
				'     (or use ←/A and →/B keys). Prev/Next to revisit.\n'
				'  4. After every collision is decided, click Done — then merge runs.\n\n'
				'Cancel anytime to abort without writing the merge.\n\n'
				'Continue to color picker and review?',
				'Review collisions',
				wx.YES_NO|wx.ICON_INFORMATION,
			)
			if review_intro.ShowModal()!=wx.ID_YES:
				review_intro.Destroy()
				return
			review_intro.Destroy()

			color_map=self._prompt_collision_outline_colors()
			review=WindowLv3_CollisionReview(
				self,
				self.folder_a,
				self.folder_b,
				self.preview.both,
				color_map,
			)
			if review.ShowModal()!=wx.ID_OK:
				review.Destroy()
				return
			collision_choices=review.get_choices()
			review.Destroy()

		progress=wx.ProgressDialog(
			'Merging datasets',
			'Starting…',
			maximum=3,
			parent=self,
			style=wx.PD_APP_MODAL|wx.PD_AUTO_HIDE,
		)
		stage={'i':0}

		def on_progress(msg):
			stage['i']=min(stage['i']+1,3)
			progress.Update(stage['i'],msg)
			wx.YieldIfNeeded()

		try:
			result=merge_datasets(
				self.folder_a,
				self.folder_b,
				self.folder_out,
				options=MergeOptions(
					collision_policy=self.collision_policy,
					collision_choices=collision_choices,
				),
				progress_callback=on_progress,
			)
		except MergeError as exc:
			progress.Destroy()
			wx.MessageBox(str(exc),'Merge failed',wx.OK|wx.ICON_ERROR)
			return
		except Exception as exc:
			progress.Destroy()
			wx.MessageBox(str(exc),'Merge failed',wx.OK|wx.ICON_ERROR)
			return

		progress.Destroy()
		collision_note=''
		if result.collisions_resolved:
			collision_note='\n  '+str(result.collisions_resolved)+' name collisions resolved ('+policy_label(result.collision_policy)+')'
		backup_note='\n  Existing annotations.json backed up.' if result.backed_up_json else ''
		wx.MessageBox(
			'Merge complete.\n'
			'  '+str(result.images_copied)+' images copied to '+result.output_path+'\n'
			'  '+str(result.annotations_written)+' annotations in annotations.json'
			+collision_note+backup_note,
			'Merge complete',
			wx.OK|wx.ICON_INFORMATION,
		)

