import random 

import numpy as np
import matplotlib.pyplot as plt

import hri_gnn.utils.visualization as visuals

from hri_gnn.config.const import Configuration

def watch_problem(problem, challenge= False, show_answer=False):
    """
    Join the RPM images and descriptors (extracted from the RAVEN Dataset annotations files) to their figures placeholders

    Parameters
    ----------
    problem: dict[str, Any], the RPM problem data : 
            - keys: 
                - 'config' : the configuration name
                - 'id' : the RPM id problem
                - 'split': the split (train, test, val)
                - 'panels': the panels images and annotations (annotations contains the descitions texts)
                - 'rules': The RPM rules 
    Returns
    -------
    None

    """
    print(f"Problem dataset: {problem['dataset']}\nProblem configuration: {problem['config']}\nProblem id: {problem['id']}\nProblem split: {problem['split']}\n")
    # Build canvas
    canvas = visuals.canvas()
    # Display Rules 
    for group_index , group in enumerate(problem['rules']):
        for rule_index, rule in enumerate(group['rules']):
            ax_canvas_rules = canvas['rules'][group_index].flatten()[rule_index]
            ax_canvas_rules.text(0.075,0.35,visuals.rule_description(rule), {"color":"b"}, fontsize=20)
            ax_canvas_rules.spines[:].set_color('orange')

    # Display Context Images 
    for ctx_panel_img_index, ctx_panel_img in enumerate(problem['panels'][0:8]): 
        ax_canvas_ctx_imgs = canvas['context'][0].flatten()[ctx_panel_img_index]
        ax_canvas_ctx_imgs.imshow(ctx_panel_img['image'], cmap='gray', vmin=0, vmax=255)
        ax_canvas_ctx_imgs.set_ylabel(ctx_panel_img_index,rotation=0, labelpad=5)

    # Display a white image of 160x160 255-colored pixels to fulfill the empty canvas
    ctx_panel_img_index = len(problem['panels'][0:8])
    image_to_draw = 255 * np.ones(shape=(160,160))
    ax_canvas_ctx_imgs = canvas['context'][0].flatten()[ctx_panel_img_index]
    ax_canvas_ctx_imgs.imshow(image_to_draw, cmap='gray', vmin=0, vmax=255)
    
    # Display Choices Images
    for chcs_panel_img_index, chcs_panel_img in enumerate(problem['panels'][8:]): 
        ax_canvas_chcs_imgs = canvas['choices'][0].flatten()[chcs_panel_img_index]
        ax_canvas_chcs_imgs.imshow(chcs_panel_img['image'], cmap='gray', vmin=0, vmax=255)
        ax_canvas_chcs_imgs.set_ylabel(chcs_panel_img_index,rotation=0, labelpad=5)

    # Display Context Descriptors 
    for ctx_panel_desc_index, ctx_panel_desc in enumerate(problem['panels'][0:8]): 
        ax_canvas_ctx_desc = canvas['context'][1].flatten()[ctx_panel_desc_index]
        ax_canvas_ctx_desc.text(0.01,0.0, visuals.panel_description(ctx_panel_desc) , {"color":"k"}, fontsize=11)
    # Display Choices Descriptors
    for chcs_panel_desc_index, chcs_panel_desc in enumerate(problem['panels'][8:]): 
        ax_canvas_chcs_desc = canvas['choices'][1].flatten()[chcs_panel_desc_index]
        ax_canvas_chcs_desc.text(0.01,0.0,visuals.panel_description(chcs_panel_desc), {"color":"k"}, fontsize=11)

    # Display Correct Answer
    if show_answer:
        _show_answer_fn(canvas, problem, challenge)
    
    if challenge:
        choice = int(input("Enter the id of your answer between [0,7]"))
        _challenge_fn(canvas, problem, choice)

def _challenge_fn(canvas, problem, choice=None):
    """
    Draw an answer (panel image and panel description) in the 9th position of the matrix)
    By default if the correct is False and the choice is None it will draw white void image of 160x160 white pixels and do not fulfill the description canvas
    Otherwise if the correct is True (wether the choice is None or not ) it will draw the correct answer
    finally if the correct is False and the choice is not None cheks if the choice is in [0,7] the draw the answer having the id = choice
    """
    print(f"Your answer is : {choice}")
    if canvas is not None:
        if choice is None:
            choice_to_draw = None
            image_to_draw = 255*np.ones(shape=(160,160))
            description_to_draw =""
        else:
            if choice < 0 or choice > 7:
                raise ValueError("Choice must be between 0 and 7")
            choice_to_draw = choice
            image_to_draw = problem['panels'][8:][choice_to_draw]['image']
            description_to_draw = visuals.panel_description(problem['panels'][8:][choice_to_draw])

        if choice_to_draw == problem['target']:
            print("Good choice !!.......")    
            canvas['context'][0].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][0].flatten()[8].spines[:].set_color(('forestgreen',1))
            canvas['context'][1].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][1].flatten()[8].spines[:].set_color(('forestgreen',1))
        else:
            print(f"Try again, hard luck !!......")
            canvas['context'][0].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][0].flatten()[8].spines[:].set_color(('red',1))
            canvas['context'][1].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][1].flatten()[8].spines[:].set_color(('red',1))

            canvas['context'][0].flatten()[8].set_ylabel(choice_to_draw,rotation=0, labelpad=5, color="red")
            canvas['context'][0].flatten()[8].yaxis.set_label_position("right")

        canvas['context'][0].flatten()[8].imshow(image_to_draw, cmap='gray', vmin=0, vmax=255)
        canvas['context'][1].flatten()[8].text(0.01,0.0,description_to_draw, {"color":"k"}, fontsize=11)

def _show_answer_fn(canvas, problem, challenge=False):
    correct_answer_index = problem['target']
    canvas['choices'][0].flatten()[correct_answer_index].spines[:].set_linewidth(4.0)
    canvas['choices'][0].flatten()[correct_answer_index].spines[:].set_color(('forestgreen',1))
    canvas['choices'][1].flatten()[correct_answer_index].spines[:].set_linewidth(4.0)
    canvas['choices'][1].flatten()[correct_answer_index].spines[:].set_color(('forestgreen',1))
    
    if not challenge:
        print(f"The correct answer is: {correct_answer_index}")
        # show correct answer image in the context canva
        ax_canvas_ctx_imgs = canvas['context'][0].flatten()[8]
        ax_canvas_ctx_imgs.spines[:].set_linewidth(4.0)
        ax_canvas_ctx_imgs.spines[:].set_color(('forestgreen',1))
        ax_canvas_ctx_imgs.imshow(problem['panels'][8:][correct_answer_index]['image'], cmap='gray', vmin=0, vmax=255)
        ax_canvas_ctx_imgs.set_ylabel(correct_answer_index,rotation=0, labelpad=5, color="forestgreen")
        ax_canvas_ctx_imgs.yaxis.set_label_position("right")

        # show correct answer description in the context canva
        ax_canvas_ctx_desc = canvas['context'][1].flatten()[8]
        ax_canvas_ctx_desc.spines[:].set_linewidth(4.0)
        ax_canvas_ctx_desc.spines[:].set_color(('forestgreen',1))
        ax_canvas_ctx_desc.text(0.01,0.0, visuals.panel_description(problem['panels'][8:][correct_answer_index]), {"color":"k"}, fontsize=11)

def plot_panel(problem, panel_id):
    panel = problem['panels'][panel_id]
    print(visuals.panel_description(panel))
    fig,ax = plt.subplots(figsize=(2,2))
    ax.imshow(panel['image'], cmap="gray")
    ax.set_title(f"\nConfig: {problem['config']}\nSplit:{problem['split']}\nProblem id:{problem["id"]}\nPanel id:{panel_id}", fontsize=9)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_random_panel(problem, seed=None):
    """
    Plot a random panel from the problem

    Args:
    ----------
    problem: dict
        The problem to plot, should have the following keys:
            'panels': list
                The list of panels
            'config': str
                The configuration name
            'split': str
                The split name
            'id': int
                The problem id
    seed: int
        The seed to use for random number generation
    """
    if seed:
        random.seed(seed)
    panel_id = random.randint(0,15)
    plot_panel(problem, panel_id)

# flat_crops, flat_attrs, flat_bboxes, flat_panel_indices, flat_batch_indices

def plot_crop(crop, attr, bbox, panel_id, config, problem_id, batch_id):
    """
    Plot a crop of a panel

    Args:
    ----------
    crop: torch.Tensor
        The crop to plot
    attr: torch.Tensor
        The attributes of the crop
    bbox: torch.Tensor
        The bounding box of the crop
    panel_id: int
        The panel id
    config: str
        The configuration name
    problem_id: int
        The problem id
    batch_id: int
        The batch id
    """
    fig,ax = plt.subplots(figsize=(2,2))
    # matplotlib (numpy based) need 
    ax.imshow(crop.permute(2,1,0), cmap="gray")
    title_parts = []
    title_parts.append(f"batch_id:{batch_id}")
    title_parts.append(f"config:{Configuration.all()[config]}")
    title_parts.append(f"problem_id:{problem_id}")
    title_parts.append(f"panel_id:{panel_id}")
    title_parts.append(f"attrs: {attr}")
    title_parts.append(f"shape:{crop.shape}")
    title_parts.append(f"bbox:{bbox}")
    ax.set_title("\n".join(title_parts), fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_panel_crops(panel_crops):
    crops = [crop['image'] for crop in panel_crops]
    fig, axes = plt.subplots(1, len(crops), figsize=(2*len(crops), 2))
    if len(crops) > 1 :
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, crop in enumerate(crops):
        axes[i].imshow(crop.permute(2,1,0), cmap="gray")
        axes[i].set_title(f"Crop {i} \nshape : {crop.shape}", fontsize=9)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

def plot_crops(crops):
    fig, axes = plt.subplots(1, len(crops), figsize=(2*len(crops), 2))
    if len(crops) > 1 :
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, crop in enumerate(crops):
        axes[i].imshow(crop.permute(2,1,0), cmap="gray")
        axes[i].set_title(f"Crop {i} \nshape : {crop.shape}", fontsize=9)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()