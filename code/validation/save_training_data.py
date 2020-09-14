import dataset.mocap as dsm


#outdir = "./log/training/exp1"
#dsm.save_recording_copy(recording=dsm.Recordings.exp1_walk1, labelsID=0, outdir=outdir)
#outdir = "./log/training/exp2"
#dsm.save_recording_copy(recording=dsm.Recordings.exp2_walk_wave1, labelsID=0, outdir=outdir)
#outdir = "./log/training/exp3"
#dsm.save_recording_copy(recording=dsm.Recordings.exp3_walk, labelsID=0, outdir=outdir)

outdir = "./log/training/exp4-put"
dsm.save_recording_copy(recording=dsm.Recordings.exp4_pass_bottle_put, labelsID=0, outdir=outdir)
outdir = "./log/training/exp4-return"
dsm.save_recording_copy(recording=dsm.Recordings.exp4_pass_bottle_put, labelsID=1, outdir=outdir)
outdir = "./log/training/exp4-hold"
dsm.save_recording_copy(recording=dsm.Recordings.exp4_pass_bottle_hold, labelsID=0, outdir=outdir)
