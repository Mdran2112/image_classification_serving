from modules.output_postprocessor.output_preproc import OutputPostprocessor, ArgMaxOutputPostprocessor


class OutputPostprocessorFactory:

    SELECTOR = {
        "argmax": ArgMaxOutputPostprocessor
    }

    @classmethod
    def get(cls, postproc_type: str, **kwargs) -> OutputPostprocessor:
        return cls.SELECTOR[postproc_type](**kwargs)