class ASRBase:
    """Base class for ASR backends."""

    sep = ""  # separator between transcript segments

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def warmup(self, audio, init_prompt=""):
        return self.transcribe(audio, init_prompt)

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")

    def set_translate_task(self):
        raise NotImplementedError("must be implemented in the child class")
    

class OnlineProcessorInterface:

    SAMPLING_RATE = 16000

    def insert_audio_chunk(self, audio):
        raise NotImplementedError("must be implemented in child class")
    
    def process_iter(self, start_time=None):
        raise NotImplementedError("must be implemented in child class")
    
    def finish(self):
        raise NotImplementedError("must be implemented in child class")