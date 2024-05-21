import cv2 as cv
import numpy as np
from parameters import *

class ImageStitcher:
    def __init__(self):
        self.images = [] # Список изображений
        self.blend = 'feather'  # Метод смешивания (блендинга) изображений
        self.blend_strength = 0  # Сила смешивания при использовании метода блендинга
        self.features = 'sift'  # Тип функций поиска ключевых точек для определения особенностей изображений
        self.match_conf = None  # Пороговое значение для отбора сопоставлений между особенностями изображений
        self.warp = 'plane'  # Тип искажения (варпинга), который будет использоваться для сопоставления изображений
        self.seam = 'no'  # Метод поиска шва между изображениями
        self.expos_comp = 'no'  # Метод компенсации экспозиции для коррекции яркости изображений
        self.wave_correct = 'no'  # Метод коррекции искажений
        
        self.try_cuda = False  # Попытка использования CUDA для ускорения операций
        self.scale_factor_work = 0.6  # Масштабный коэффициент для работы с изображениями
        self.matcher = 'homography'  # Тип метода сопоставления особенностей между изображениями 
        self.estimator = 'homography'  # Тип оценщика параметров проецирования для оценки матрицы гомографии и параметров искажения
        self.conf_thresh = 1.0  # Пороговое значение для коррекции параметров проецирования
        self.ba = 'ray'  # Тип метода минимизации стоимости пакета для коррекции параметров проецирования
        self.ba_refine_mask = 'xxxxx'  # Маска для определения, какие параметры проецирования следует корректировать
        self.scale_factor_seam = 0.1  # Масштабный коэффициент для шва изображений
        self.compose_megapix = -1  # Мегапиксельная величина для масштабирования изображений при компоновке
        self.expos_comp_nr_feeds = 1  # Количество проходов для метода компенсации экспозиции
        self.expos_comp_nr_filtering = 2  # Количество фильтруемых кадров для метода компенсации экспозиции
        self.expos_comp_block_size = 32  # Размер блока для метода компенсации экспозиции
        self.rangewidth = -1  # Ширина диапазона сопоставления между изображениями
    
    def set_images(self, images):
        self.images = images

    def set_features(self, features):
        self.features = features
    
    def set_blend(self, blend):
        self.blend = blend

    def set_blend_strength(self, blend_strength):
        self.blend_strength = blend_strength
            
    def set_match_conf(self, match_conf):
        self.match_conf = match_conf

    def set_warp(self, warp):
        self.warp = warp

    def set_seam(self, seam):
        self.seam = seam
            
    def set_expos_comp(self, expos_comp):
        self.expos_comp = expos_comp
                
    def set_wave_correct(self, wave_correct):
        self.wave_correct = wave_correct
    
    def set_try_cuda(self, try_cuda):
        self.try_cuda = try_cuda
        
    def set_work_megapix(self, work_megapix):
        self.scale_factor_work = work_megapix
        
    def set_matcher(self, matcher):
        self.matcher = matcher
        
    def set_estimator(self, estimator):
        self.estimator = estimator
        
    def set_conf_thresh(self, conf_thresh):
        self.conf_thresh = conf_thresh
        
    def set_ba(self, ba):
        self.ba = ba
        
    def set_ba_refine_mask(self, ba_refine_mask):
        self.ba_refine_mask = ba_refine_mask
        
    def set_seam_megapix(self, seam_megapix):
        self.scale_factor_seam = seam_megapix
        
    def set_compose_megapix(self, compose_megapix):
        self.compose_megapix = compose_megapix
        
    def set_expos_comp_nr_feeds(self, expos_comp_nr_feeds):
        self.expos_comp_nr_feeds = expos_comp_nr_feeds
        
    def set_expos_comp_nr_filtering(self, expos_comp_nr_filtering):
        self.expos_comp_nr_filtering = expos_comp_nr_filtering
        
    def set_expos_comp_block_size(self, expos_comp_block_size):
        self.expos_comp_block_size = expos_comp_block_size
        
    def set_output(self, output):
        self.output = output
        
    def set_rangewidth(self, rangewidth):
        self.rangewidth = rangewidth

    def get_matcher(self):
        try_cuda = self.try_cuda
        matcher_type = self.matcher
        if self.match_conf is None:
            if self.features == 'orb':
                match_conf = 0.3
            else:
                match_conf = 0.65
        else:
            match_conf = self.match_conf
        range_width = self.rangewidth
        if matcher_type == "affine":
            matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
        elif range_width == -1:
            matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
        else:
            matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
        return matcher

    def get_compensator(self):
        expos_comp_type = EXPOS_COMP_CHOICES[self.expos_comp]
        expos_comp_nr_feeds = self.expos_comp_nr_feeds
        expos_comp_block_size = self.expos_comp_block_size
        if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
            compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
        elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv.detail_BlocksChannelsCompensator(
                expos_comp_block_size, expos_comp_block_size,
                expos_comp_nr_feeds
            )
        else:
            compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
        return compensator

    def load_images(self, img_names):
        images = []
        for name in img_names:
            img = cv.imread(name)
            if img is None:
                raise ValueError("Cannot read image ", name)
            images.append(img)
        self.images = images

    def stitch_images(self):

        finder = FEATURES_FIND_CHOICES[self.features]()
        
        seam_work_aspect = 1
        full_img_sizes = []
        features = []
        images = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False

        for img in self.images:
            # Считывание изображений и определение их размеров
            full_img_sizes.append((img.shape[1], img.shape[0]))

            # Если масштаб работы не задан, используем исходное изображение
            if self.scale_factor_work < 0:
                new_img = img
                work_scale = 1
                is_work_scale_set = True
            else:
                # Вычисление масштаба работы на основе мегапикселей
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(self.scale_factor_work * 1e6 / (img.shape[0] * img.shape[1])))
                    is_work_scale_set = True
                new_img = cv.resize(src=img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
            
            # Вычисление масштаба шва на основе мегапикселей
            if is_seam_scale_set is False:
                if self.scale_factor_seam > 0:
                    seam_scale = min(1.0, np.sqrt(self.scale_factor_seam * 1e6 / (img.shape[0] * img.shape[1])))
                else:
                    seam_scale = 1.0
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            
            # Вычисление особенностей изображения
            img_feat = cv.detail.computeImageFeatures2(finder, new_img)
            features.append(img_feat)
            
            # Масштабирование изображения для шва
            new_img = cv.resize(src=img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            images.append(new_img) 

        # Сопоставление признаков
        matcher = self.get_matcher()
        p = matcher.apply2(features)
        matcher.collectGarbage()

        # Оставление самой большой компоненты соответствий
        indices = cv.detail.leaveBiggestComponent(features, p, self.conf_thresh)
        img_subset = []
        images_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            images_subset.append(self.images[indices[i]])
            img_subset.append(images[indices[i]])
            full_img_sizes_subset.append(full_img_sizes[indices[i]])
        images = img_subset
        imgs = images_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(imgs)
        if num_images < 2:
            raise ValueError("Need more images")

        # Оценка параметров проецирования
        estimator = ESTIMATOR_CHOICES[self.estimator]()
        b, projection_params = estimator.apply(features, p, None)
        if not b:
            raise ValueError("Homography estimation failed.")
        for params in projection_params:
            params.R = params.R.astype(np.float32)

        # Коррекция параметров проецирования
        adjuster = BA_COST_CHOICES[self.ba]()
        adjuster.setConfThresh(self.conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if self.ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if self.ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if self.ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if self.ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if self.ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, projection_params = adjuster.apply(features, p, projection_params)
        if not b:
            raise ValueError("Camera parameters adjusting failed.")

        # Вычисление масштаба для трансформации
        focals = []
        for params in projection_params:
            focals.append(params.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        
        # Коррекция волнового эффекта
        if self.wave_correct is not None:
            rmats = []
            for params in projection_params:
                rmats.append(np.copy(params.R))
            rmats = cv.detail.waveCorrect(rmats, WAVE_CORRECT_CHOICES[self.wave_correct])
            for idx, params in enumerate(projection_params):
                params.R = rmats[idx]
        
        # Подготовка масок для изображений
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        # Применение преобразований к изображениям и маскам
        warper = cv.PyRotationWarper(self.warp, warped_image_scale * seam_work_aspect)
        for idx in range(0, num_images):
            K = projection_params[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(images[idx], K, projection_params[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, projection_params[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())

        # Преобразование изображений в float
        images_warped_f = []
        for new_img in images_warped:
            imgf = new_img.astype(np.float32)
            images_warped_f.append(imgf)

        # Применение компенсации экспозиции
        compensator = self.get_compensator()
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        # Поиск шва между изображениями
        seam_finder = SEAM_FIND_CHOICES[self.seam]
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)

        compose_scale = 1
        corners = []
        sizes = []
        blender = None

        for idx, img in enumerate(self.images):
            # Если масштаб композиции не установлен, вычисляем его на основе указанных мегапикселей
            if not is_compose_scale_set:
                if self.compose_megapix > 0:
                    # Вычисление масштаба композиции для изображения
                    compose_scale = min(1.0, np.sqrt(self.compose_megapix * 1e6 / (img.shape[0] * img.shape[1])))
                is_compose_scale_set = True
                
                # Вычисление масштаба работы для композиции
                compose_work_aspect = compose_scale / work_scale
                # Обновление масштаба изображений, скорректированных для композиции
                warped_image_scale *= compose_work_aspect
                # Создание объекта-варпера с учетом масштаба
                warper = cv.PyRotationWarper(self.warp, warped_image_scale)

                # Применение масштаба к параметрам камеры
                for i in range(0, len(images)):
                    projection_params[i].focal *= compose_work_aspect
                    projection_params[i].ppx *= compose_work_aspect
                    projection_params[i].ppy *= compose_work_aspect
                    sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                        int(round(full_img_sizes[i][1] * compose_scale)))
                    K = projection_params[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, projection_params[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            
            # Изменение размера изображения, если необходимо, на основе масштаба композиции
            if abs(compose_scale - 1) > 1e-1:
                new_img = cv.resize(src=img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            else:
                new_img = img
            _img_size = (new_img.shape[1], new_img.shape[0]) # Размеры изображения

            # Выполнение искажения изображения с использованием параметров камеры и варпера
            K = projection_params[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(new_img, K, projection_params[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            
            # Создание маски для искаженного изображения
            mask = 255 * np.ones((new_img.shape[0], new_img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, projection_params[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            
            # Применение компенсатора к искаженному изображению
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            
            # Расширение маски шва и объединение ее с искаженной маской
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            
            # Смешивание искаженного изображения с предыдущими с использованием блендера
            if blender is None:
                # Создание блендера с параметрами смешивания
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.blend == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                elif self.blend == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
        
        # Сохранение итоговой панорамы
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        if result.dtype != np.uint8:
            result = cv.convertScaleAbs(result)

        return result  