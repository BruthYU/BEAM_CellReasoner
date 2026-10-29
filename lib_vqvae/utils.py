import torch
import os


def save_model_and_results(model, eval_adata, eval_loader, results, hyperparameters, timestamp, args, training_state=None):
    # saving model
    SAVE_MODEL_PATH = os.path.join(args.outdir, f'epoch_{timestamp}')
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters}
    if training_state is not None:
        results_to_save.update(training_state)
    if "config" in hyperparameters:
        results_to_save["config"] = hyperparameters["config"]
    destination = os.path.join(SAVE_MODEL_PATH, 'model.pth')
    temporary = destination + '.tmp'
    torch.save(results_to_save, temporary)
    os.replace(temporary, destination)
