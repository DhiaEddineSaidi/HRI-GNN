#from IPython.display import display, clear_output
#import matplotlib.pyplot as plt

import numpy as np
from hri_gnn.utils.visuals import rpm_canvas

def rpm_oracle(rpm_data, return_canvas= False, show_answer=False)-> dict | None:
    """
    Join the RPM images and descriptors (extracted from the RAVEN Dataset annotations files) to their figures placeholders

    Parameters
    ----------
    canvas: dict{'str':ndarray(Axes)}
        The placeholders canvas to draw in
            * keys:
                -'rules': [first_components_rules_axes, second_component_rules_axes] , 
                'context': [context_images_axes, context_descriptors_axes], 
                'choices': [choices_images_axes, choices_descriptor_axess]
    rpm_data: dict[str, Any]
        the RPM data : 
            * keys: 
                - 'config' : the configuration name
                - 'id' : the RPM id problem
                - 'split': the split (train, test, val)
                - 'panels': the panels images and annotations (annotations contains the descitions texts)
                - 'rules': The RPM rules 
    Returns
    -------
    None

    """
    # Build canvas
    canvas = rpm_canvas()
    # Display Rules 
    for group_index , group in enumerate(rpm_data['rules']):
        for rule_index, rule in enumerate(group['rules']):
            ax_canvas_rules = canvas['rules'][group_index].flatten()[rule_index]
            ax_canvas_rules.text(0.075,0.35,rule['description'], {"color":"b"}, fontsize=20)
            ax_canvas_rules.spines[:].set_color('orange')

    # Display Context Images 
    for ctx_panel_img_index, ctx_panel_img in enumerate(rpm_data['panels'][0:8]): 
        ax_canvas_ctx_imgs = canvas['context'][0].flatten()[ctx_panel_img_index]
        ax_canvas_ctx_imgs.imshow(ctx_panel_img['image'], cmap='gray', vmin=0, vmax=255)
        ax_canvas_ctx_imgs.set_ylabel(ctx_panel_img_index,rotation=0, labelpad=5)

    # Display a white image of 160x160 255-colored pixels to fulfill the empty canvas
    ctx_panel_img_index = len(rpm_data['panels'][0:8])
    image_to_draw = 255 * np.ones(shape=(160,160))
    ax_canvas_ctx_imgs = canvas['context'][0].flatten()[ctx_panel_img_index]
    ax_canvas_ctx_imgs.imshow(image_to_draw, cmap='gray', vmin=0, vmax=255)
    
    # Display Choices Images
    for chcs_panel_img_index, chcs_panel_img in enumerate(rpm_data['panels'][8:]): 
        ax_canvas_chcs_imgs = canvas['choices'][0].flatten()[chcs_panel_img_index]
        ax_canvas_chcs_imgs.imshow(chcs_panel_img['image'], cmap='gray', vmin=0, vmax=255)
        ax_canvas_chcs_imgs.set_ylabel(chcs_panel_img_index,rotation=0, labelpad=5)

    # Display Context Descriptors 
    for ctx_panel_desc_index, ctx_panel_desc in enumerate(rpm_data['panels'][0:8]): 
        ax_canvas_ctx_desc = canvas['context'][1].flatten()[ctx_panel_desc_index]
        ax_canvas_ctx_desc.text(0.01,0.0,ctx_panel_desc['annotation']['description'], {"color":"k"}, fontsize=11)
    # Display Choices Descriptors
    for chcs_panel_desc_index, chcs_panel_desc in enumerate(rpm_data['panels'][8:]): 
        ax_canvas_chcs_desc = canvas['choices'][1].flatten()[chcs_panel_desc_index]
        ax_canvas_chcs_desc.text(0.01,0.0,chcs_panel_desc['annotation']['description'], {"color":"k"}, fontsize=11)
    

    # Display Correct Answer
    if show_answer:
        correct_answer_index = rpm_data['target'] # Canvas choices contains 9 positions and not 8 tha's why we substract -9
        print(f"The correct answer is: {correct_answer_index}")
        canvas['choices'][0].flatten()[correct_answer_index].spines[:].set_linewidth(4.0)
        canvas['choices'][0].flatten()[correct_answer_index].spines[:].set_color(('forestgreen',1))

        canvas['choices'][1].flatten()[correct_answer_index].spines[:].set_linewidth(4.0)
        canvas['choices'][1].flatten()[correct_answer_index].spines[:].set_color(('forestgreen',1))

        ax_canvas_ctx_imgs = canvas['context'][0].flatten()[8]
        ax_canvas_ctx_imgs.spines[:].set_linewidth(4.0)
        ax_canvas_ctx_imgs.spines[:].set_color(('forestgreen',1))
        ax_canvas_ctx_imgs.imshow(rpm_data['panels'][8:][correct_answer_index]['image'], cmap='gray', vmin=0, vmax=255)
        ax_canvas_ctx_imgs.set_ylabel(correct_answer_index,rotation=0, labelpad=5, color="forestgreen")
        ax_canvas_ctx_imgs.yaxis.set_label_position("right")

        ax_canvas_ctx_desc = canvas['context'][1].flatten()[8]
        ax_canvas_ctx_desc.spines[:].set_linewidth(4.0)
        ax_canvas_ctx_desc.spines[:].set_color(('forestgreen',1))
        ax_canvas_ctx_desc.text(0.01,0.0,rpm_data['panels'][8:][correct_answer_index]['annotation']['description'], {"color":"k"}, fontsize=11)

        
    #ax_canvas_ctx_imgs.set_title()
    #figure = canvas['figure']
    #display(figure)
    
    if return_canvas:
        return canvas


def pick_rpm_answer(rpm_data, choice=None, show_answer=False):
    """
    Draw an answer (panel image and panel description) in the 9th position of the matrix)
    By default if the correct is False and the choice is None it will draw white void image of 160x160 white pixels and do not fulfill the description canvas
    Otherwise if the correct is True (wether the choice is None or not ) it will draw the correct answer
    finally if the correct is False and the choice is not None cheks if the choice is in [0,7] the draw the answer having the id = choice
    """

    # Build rpm 
    canvas = rpm_oracle(rpm_data, return_canvas=True,show_answer=show_answer)
    if canvas is not None:
        if choice is None:
            choice_to_draw = None
            image_to_draw = 255*np.ones(shape=(160,160))
            description_to_draw =""
        else:
            if choice < 0 or choice > 8:
                raise ValueError("Choice must be between 0 and 8")
            choice_to_draw = choice
            image_to_draw = rpm_data['panels'][8:][choice_to_draw]['image']
            description_to_draw = rpm_data['panels'][8:][choice_to_draw]['annotation']['description']

        if choice_to_draw == rpm_data['target']:
            print("Good choice !!.......")    
            canvas['context'][0].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][0].flatten()[8].spines[:].set_color(('forestgreen',1))
            canvas['context'][1].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][1].flatten()[8].spines[:].set_color(('forestgreen',1))
        else:
            print(f"Hard luck !!......")
            canvas['context'][0].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][0].flatten()[8].spines[:].set_color(('red',1))
            canvas['context'][1].flatten()[8].spines[:].set_linewidth(4.0)
            canvas['context'][1].flatten()[8].spines[:].set_color(('red',1))

            canvas['context'][0].flatten()[8].set_ylabel(choice_to_draw,rotation=0, labelpad=5, color="red")
            canvas['context'][0].flatten()[8].yaxis.set_label_position("right")

        canvas['context'][0].flatten()[8].imshow(image_to_draw, cmap='gray', vmin=0, vmax=255)
        canvas['context'][1].flatten()[8].text(0.01,0.0,description_to_draw, {"color":"k"}, fontsize=11)
      
        #figure = canvas['figure']
        #clear_output(wait=True)
        #display(figure)
        #plt.close(figure)
        #figure.canvas.draw()
        #figure.canvas.flush_events()