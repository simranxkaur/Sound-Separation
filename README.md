# Sound-Separation

A python script that can separate the individual sound events in a polyphonic audio clip. The middle 1-second, at a sample rate of 44.1kHz, of each sound event is returned which in turn populate  training, testing, and validation datasets for machine learning. Each dataset example creates anywhere from 1-4 Tag clips depending on the number of sound events.
