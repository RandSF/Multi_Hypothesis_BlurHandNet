
def get_runner(name, opt, args, logger, training):
    if name == 'runner_bh':
        from training.runner_bh import Runner
        return Runner(opt, args, logger, training)
    if name == 'runner_bh_rm':
        from training.runner_bh_rm import Runner
        return Runner(opt, args, logger, training)
    raise KeyError

def get_trainer(name, opt, training):
    if name == 'training_model':
        from training.training_model import TrainingModel
        return TrainingModel(opt, training=training)
    if name == 'training_model_rm':
        from training.training_model_rm import TrainingModel
        return TrainingModel(opt, training=training)
    raise KeyError