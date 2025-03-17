import os
import torch
from sd3_inferer import CONFIGS, SD3Inferencer
from tqdm import tqdm

prompts = ["photo of group 19th century cult cultists in the dark forest by Diane Arbus and Louis Daguerre",
            "attractive justin bieber as a god. highly detailed painting by gaston bussiere, craig mullins, j. c. leyendecker 8 k",
            "bill clinton wearing blue dress",
            "portrait of Huntress Wizard from adventure time, in the style of adventure time, ultra detailed 4k, beautiful inking lines, stunning gradient colors, trending on artstation, digital illustration, character design, concept art",
            "symbol representation of an object",
            "a nerd making ai art",
            "The last selfie found on earth after the apocalypse",
            "a colorful comic noir illustration painting of a cyberpunk skull by sachin teng and sergey kolesov and ruan jia and heng z. graffiti art, sci fi, hyper detailed. octane render. trending on artstation",
            "img _ 2 4 2 4. jpg. jesse pinkman selfie. candid, snapchat, instagram, hashtags, accidental selfie, front camera, tiktok, popular, realistic, real life",
            "symmetry! human cell, intricate, elegant, highly detailed, concept art, smooth, sharp focus, lineart, illustration,, penned with black on white, 8 k",
            "Obama caught on trail cam footage, creepy distorted night cam",
            "Ellen DeGeneres and Jonah Hill Cisco WebEx Zoom Microsoft Teams Google Hangouts Facetime Video Call Meeting",
            "pete davidson stuck inside a bubble",
            "human sculpture by Anna Rubincam, masterpiece",
            "Artwork by John Howe of Abe the Forgotten Beast, A towering humanoid composed of rose gold, with a gaunt appearance and a matted grey fur",
            "portrait of joe biden or golum by greg ruthkowski",
            "a realistic detailed portrait painting of a monster by john kenn mortensen, santiago caruso, synthwave cyberpunk psychedelic vaporwave",
            "Portrait of George Soros as the emperor palpatine from star wars, made by stanly artgerm lau, wlop, rossdraws, james jean, andrei riabovitchev ,marc simonetti",
            "Liz Truss looking sad, political cartoon, style of Ralph Steadman",
            "malakai black in sleepy hollow, full body, big two toned eyes, teeth gritted, horror, intricate details, cinematic, epic, realistic, anatomy, tomer hanuka, uplight, artstation, photorealistic, scary",
            "super rare pepe kek with feels good pepe frog meme kek memetic algae fractal kek chaos super rare pepe meme frog kek",
            "angler fish skeleton",
            "Shaking hands, vertical symmetry, close up shot, detailed hands, beautiful moody artwork by Greg Rutkowski and Asher Duran",
            "cute ghost holding a balloon, anime, cartoon",
            "old photograph of a victorian child surrounded by glowing shadow eyes, demons, ghosts, paranormal evidence in the background",
            "a horrifying eldritch man by Beksinski and Junji Ito",
            "kerli koiv in sleepy hollow, full body, big two toned eyes, teeth gritted, horror, intricate details, cinematic, epic, realistic, anatomy, tomer hanuka, uplight, artstation, photorealistic, scary",
            "evil sorceress in the style of adventure time, hd, trending on artstation, digital illustration, cartoon character design, concept art",
            "hyperrealistic close up studio portrait of aging old Robert Pattinson age 85 wrinkled frowning, oil painting by Ivan Albright and Edward Hopper and Ron Mueck, trending on artstation Studio lighting hyperrealism",
            "( a death metal album cover ) ( ( of donald trump ) with laser eyes ) ( trending on artstation )",
            "painting of a boy back pose",
            "blurry film still of obama meeting gus fring at los pollos hermanos, breaking bad scene, octane render, 4k, photorealistic, detailed",
            "tank girl sitting alone on top of the tank, holding a granite luncher. smiling. laurie greasley,",
            "death metal album cover,",
            "yuki   onna haunting an abandoned asylum by zdislaw beksinski, heavy fog, vibrant and creepy atmosphere, bones cover the floor, style of takato yamamoto and vania zouravliov",
            "detailed photo of beautiful woman statue, various lacivious seducing poses, photorealism, intricate detail, a few light reflexions, museum diffuse lighting",
            "photorealistic picture of Putin and Viktor orban kissing by Mandy Jurgens and Richard Schmid and chuck close and mucha",
            "simple primitive tube shape, textured with photorealistic human skin, photoreal details, straight smooth vertical, highly realistic bump map, surface painter, pixar renderman, metahuman",
            "painting of barack obama in the place of the mona lisa in the mona lisa painting",
            "dinosaur wearing a suit",
            "pepe the frog having a talk with Donald Trump, ultra realistic",
            "crazy asian man in the style of lord of the ring by rossdraws, artgrem",
            "propaganda poster justin trudeau as ruler of north korea, 8 k, trending on artstation",
            "A young woman with big round green eyes, medium length wavy blonde brown hair, rectangular face, masculine facial features, wearing a seafoam t shirt, HD selfie",
            "stalin eat kids, children, hohol anatomic shot",
            "portrait of Aphrodite and Donald Trump, photorealistic, 4K",
            "grindcore band logo, metal font, unreadable",
            "very very very perfect face of a woman, soft skin, shy smile",
            "cursed illustration of beautifully ominous creepy underworld lovecraftian entity peeking into 1 9 8 0's teenagers bedroom. manga style of kentaro mirua, by norman rockwell, weirdcore",
            "Pepe the Frog in the parade, Socialist realism, North Korea propaganda style"]

# Actual model file path
MODEL = "/home/lxc/cache/AI-ModelScope/stable-diffusion-3___5-large/sd3.5_large.safetensors"
# TextEncoder MODEL FOLDER
MODEL_FOLDER = "/home/lxc/cache/AI-ModelScope/stable-diffusion-3___5-large/text_encoders"

OUTDIR = "/home/lxc/sd3.5/outputs/lexica"

WIDTH = 1024
HEIGHT = 1024
SEED = 23

def get_empty_latent(batch_size, width, height, seed, device):
    shape = (batch_size, 16, height // 8, width // 8)
    latents = torch.zeros(shape, device=device)
    for i in range(shape[0]):
        prng = torch.Generator(device=device).manual_seed(int(seed + i))
        latents[i] = torch.randn(shape[1:], generator=prng, device=device)
    return latents

if __name__ == "__main__":
    config = CONFIGS["sd3.5_large"]
    _shift = config["shift"]
    _steps = config["steps"]
    _cfg = config["cfg"]
    _sampler = config["sampler"]

    os.makedirs(OUTDIR, exist_ok=True)

    with torch.no_grad():
        inferencer = SD3Inferencer()
        inferencer.load(MODEL, MODEL_FOLDER, _shift, text_encoder_device="cuda")
        pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        for i, prompt in pbar:
            for j in range(10):
                init_latent = get_empty_latent(1, WIDTH, HEIGHT, SEED, "cuda")
                seed_num = torch.randint(0, 100000, (1,)).item()

                image = inferencer.s_gen_image(prompt, init_latent, seed_num, _steps, _cfg)

                save_path = os.path.join(OUTDIR, f"{i+51}-{j+1}.png")
                image.save(save_path)
        print("Done")
