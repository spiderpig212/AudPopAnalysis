import numpy as np

locLabels = ['brainSurface', 'bottomSoundResp']

# Order is subject: session: [surfaceChannel, TeA/Bottom Channel], Chosen by when two channels near each other have at
# least 3 SR neurons. Brain surface was estimated off the 15-20 ms line in LFP figs and confirmed with 20-30 ms line
probeLocation = {
    'feat014': {'2024-03-16': [260, 66],  # 42 shows first sound responsive
                '2024-03-18': [260, 62],
                '2024-03-19': [270, 53],  # 38 shows first round of SR neurons
                },
    'feat015': {'2024-03-20': [230, 109],  # Very limited data makes meeting above criteria hard for this animal
                '2024-03-21': [280, 137],
                '2024-03-22': [310, 159],
                },
    'feat016': {'2024-03-21': [290, 150],  # Technically doesn't meet the criteria above for the bottom, but the SR is very strong
                '2024-03-22': [300, np.nan],  # Surface is a bit questionable. No neurons seem sound responsive. Makes sense with histology
                '2024-03-23': [280, 111],  # First SR at 33
                '2024-03-24': [260, 94],  # There is a putative cell at 292, so surface may be off. Only has single spike
                '2024-04-04': [270, 102],  # LFP is hard to tell for out of brain. First SR at 46. Mostly silenced in response to stim
                '2024-04-08': [280, 113],
                '2024-04-09': [300, 182],  # First SR at 118
                '2024-04-10': [260, 132],  # First SR at 2
                '2024-04-11': [240, 150],  # First SR at 150. Cannot meet criterion above. Also hard to estimate surface
                '2024-04-12': [250, 142],  # First SR at 142. Cannot meet criterion above. Also hard to estimate surface
                '2024-04-17': [280, 125],  # First SR at 114
                },
    'feat018': {'2024-06-06': [260, 141],  # There is a bump that happens again for surface, so may be ~320
                '2024-06-07': [280, 76],  # First sound response at 69
                '2024-06-10': [270, 121],  # First sound response at 97. Surface is off 20-30 ms line as 15-20 is weird
                '2024-06-11': [200, 162],  # Surface should be ~200, but putative cells still found. First SR at 118
                '2024-06-12': [260, 122],  # First SR at 89
                '2024-06-14': [250, 41],
                '2024-06-15': [250, 38],  # First SR at 25
                '2024-06-17': [300, 73],  # First SR at 57
                '2024-06-18': [200, 68],  # First SR at 9, maybe even 0
                '2024-06-26': [250, 136],
                '2024-06-27': [280, 105],  # First SR at 100
                }
}