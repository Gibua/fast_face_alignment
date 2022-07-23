class FastFAN(nn.Module):

	def __init__(self, num_modules = 1,depth = 2,imp = 0,device = 0):
		super(FastFAN, self).__init__()
		self.batch_size =1

		self.imp=imp
		self.device = device
		self.num_modules = num_modules

		# Base part
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = ConvBlock(64, 128)
		self.conv3 = ConvBlock(128, 128)
		self.conv4 = ConvBlock(128, 256)

		# Stacking part
		for hg_module in range(self.num_modules):
			self.add_module('m' + str(hg_module), HourGlass(1, depth, 256))
			self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
			self.add_module('conv_last' + str(hg_module),
							nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
			self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
			if hg_module == self.num_modules - 1 and imp!=0:
				self.add_module('leasy' + str(hg_module), nn.Conv2d(256, imp, kernel_size=1, stride=1, padding=0))
				self.add_module('lhard' + str(hg_module),
								nn.Conv2d(256+imp, 68-imp, kernel_size=1, stride=1, padding=0))
			else:
				self.add_module('l' + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0))


			if hg_module < self.num_modules - 1:
				self.add_module(
					'bl' + str(hg_module), nn.Conv2d(self.lastFnum, 256, kernel_size=1, stride=1, padding=0))
				self.add_module('al' + str(hg_module), nn.Conv2d(68,
																 256, kernel_size=1, stride=1, padding=0))



	def getLandmarksFromFrame(self,image,detected_faces):
		if len(detected_faces) == 0:
			return []
		centers = []
		scales = []
		for i, d in enumerate(detected_faces):
			center = torch.FloatTensor(
				[d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
			center[1] = center[1] - (d[3] - d[1]) * 0.12
			scale = (d[2] - d[0] + d[3] - d[1]) / 195
			centers.append(center)
			scales.append(scale)
			inp = crop(image, center, scale)
			inp = torch.from_numpy(inp.transpose(
				(2, 0, 1))).float()
			inp = inp.to(self.device)
			inp.div_(255.0).unsqueeze_(0)
			if i == 0:
				imgs = inp
			else:
				imgs = torch.cat((imgs,inp), dim=0)

		out = self.forward(imgs)
		out = out[-1].cpu()
		pts, pts_img = get_preds_fromhm(out, centers, scales)
		#pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
		#landmarks.append(pts_img.numpy())
		return pts_img.numpy().tolist()

	def forward(self, x):

		x = F.relu(self.bn1(self.conv1(x)), True)
		x = F.avg_pool2d(self.conv2(x), 2, stride=2)
		x = self.conv3(x)
		x = self.conv4(x)

		previous = x

		outputs = []
		for i in range(self.num_modules):
			hg = self._modules['m' + str(i)](previous)
			ll = hg
			ll = self._modules['top_m_' + str(i)](ll)
			ll = self._modules['conv_last' + str(i)](ll)
			ll = self._modules['bn_end' + str(i)](ll)
			ll = F.relu(ll, True)


			# Predict heatmaps
			if i == (self.num_modules-1) and self.imp!=0:
				easyp = self._modules['leasy' + str(i)](ll)
				hardp = self._modules['lhard' + str(i)](torch.cat((ll,easyp),1))
				tmp_out = torch.cat((easyp,hardp),1)
				outputs.append(tmp_out)
			else:
				tmp_out = self._modules['l' + str(i)](ll)
				outputs.append(tmp_out)

			if i < self.num_modules - 1:
				ll_ = self._modules['bl' + str(i)](ll)
				tmp_out_ = self._modules['al' + str(i)](tmp_out)
				previous = previous + ll_ + tmp_out_

		return outputs
