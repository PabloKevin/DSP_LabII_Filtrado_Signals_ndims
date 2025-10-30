import simpleaudio as sa

wave_obj = sa.WaveObject.from_wave_file("DSP_LabII_Filtrado_Signals_ndims/audios/violin_cut.wav")
play_obj = wave_obj.play()
play_obj.wait_done()
