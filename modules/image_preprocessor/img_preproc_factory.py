from modules.image_preprocessor.img_preprocessor import ImagePreprocessor, NormAndPaddingImagePreprocessor, NoPreproc


class ImagePreprocessorFactory:

    SELECTOR = {
        "normalize_and_pad": NormAndPaddingImagePreprocessor,
        "no_prepro": NoPreproc
    }

    @classmethod
    def get(cls, preproc_type: str, **kwargs) -> ImagePreprocessor:
        return cls.SELECTOR[preproc_type](**kwargs)


