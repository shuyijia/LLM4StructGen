from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from llm4structgen.zmatrix import zmatrix2struct

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")

def generate(model, tokenizer, batch, temperature, top_p, max_new_tokens=512):
    """
    generate a batch of structures
    """
    model.eval()
    generated_ids = model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature, 
        top_p=top_p, 
    )

    gen_strs = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return gen_strs

def unconditional_sampling(model, tokenizer, temperature, top_p, n_tries=3):
    """
    generate a single structure unconditionally
    """
    prompt = (
        'Below is a description of a bulk material. '
        'Generate a description of the lengths and angles '
        'of the lattice vectors and then the element type '
        'and coordinates for each atom within the lattice:\n'
    )

    # batch = tokenizer([prompt], return_tensors="pt")
    batch = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}

    i = 0
    while i < n_tries:
        gen_strs = generate(model, tokenizer, batch, temperature, top_p)
        assert len(gen_strs) == 1
        gen_str = gen_strs[0]

        material_str = gen_str.replace(prompt, "")
        try:
            cif_str = parse_fn(material_str)
            _ = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            i += 1
            continue

        return {
            "temperature": temperature,
            "top_p": top_p,
            "cif": cif_str,
        }
    
    raise KeyError(f"Failed to generate valid structure after {n_tries} tries")

def unconditional_zmatrix_sampling(model, tokenizer, temperature, top_p, n_tries=3):
    """
    generate a single structure unconditionally
    """
    prompt = (
        'Below is a description of a bulk material where each atom is described by its element type and three attributes: '
        '1. distance to the previous atom, '
        '2. angle to the previous two atoms, '
        '3. dihedral angle to the previous three atoms. '
        'The first three Fm atoms are dummies that help define the rest of the material. '
        'Generate a description of the lengths and angles of the lattice vectors and the three dummy Fm atoms, '
        'followed by the element type and the three attributes for each atom within the lattice:\n'
    )

    # batch = tokenizer([prompt], return_tensors="pt")
    batch = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}

    i = 0
    while i < n_tries:
        gen_strs = generate(model, tokenizer, batch, temperature, top_p)
        assert len(gen_strs) == 1
        gen_str = gen_strs[0]

        material_str = gen_str.replace(prompt, "")
        try:
            cif_str = zmatrix2struct(material_str)
            _ = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            i += 1
            continue

        return {
            "temperature": temperature,
            "top_p": top_p,
            "cif": cif_str,
        }
    
    raise KeyError(f"Failed to generate valid structure after {n_tries} tries")


def conditional_composition(model, tokenizer, composition, temperature, top_p, n_tries=3):
    """
    generate a single structure given composition
    """
    prompt = (
        'Below is a description of a bulk material. '
        'The chemical formula is <composition>. '
        'Generate a description of the lengths and angles '
        'of the lattice vectors and then the element type '
        'and coordinates for each atom within the lattice:\n'
    )

    prompt = prompt.replace("<composition>", composition)
    # batch = tokenizer([prompt], return_tensors="pt")
    batch = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}

    i = 0
    while i < n_tries:
        gen_strs = generate(model, tokenizer, batch, temperature, top_p)
        assert len(gen_strs) == 1
        gen_str = gen_strs[0]

        material_str = gen_str.replace(prompt, "")
        try:
            cif_str = parse_fn(material_str)
            _ = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            i += 1
            continue

        return {
            "temperature": temperature,
            "top_p": top_p,
            "composition": composition,
            "cif": cif_str,
        }
    
    raise KeyError(f"Failed to generate valid structure after {n_tries} tries")

def batch_conditional_composition(model, tokenizer, compositions, batch_size, temperature, top_p, n_tries=3):
    """
    generate a batch of structures given compositions
    """
    batch_size = min(batch_size, len(compositions))

    base_prompt = (
        'Below is a description of a bulk material. '
        'The chemical formula is <composition>. '
        'Generate a description of the lengths and angles '
        'of the lattice vectors and then the element type '
        'and coordinates for each atom within the lattice:\n'
    )

    prompts = [base_prompt.replace("<composition>", x) for x in compositions]
    batch_output = []

    # batch generation
    for i in range(0, len(prompts), batch_size):
        batch_compositions = compositions[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        # batch = tokenizer(batch_prompts, return_tensors="pt")
        batch = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.cuda() for k, v in batch.items()}

        for _ in range(n_tries):
            minibatch_output = []
            gen_strs = generate(model, tokenizer, batch, temperature, top_p)
            for composition, prompt, gen_str in zip(batch_compositions, batch_prompts, gen_strs):
                material_str = gen_str.replace(prompt, "")
                try:
                    cif_str = parse_fn(material_str)
                    _structure = Structure.from_str(cif_str, fmt="cif")
                except Exception as e:
                    break

                minibatch_output.append({
                    "temperature": temperature,
                    "top_p": top_p,
                    "composition": composition,
                    "cif": cif_str,
                })
            if len(minibatch_output) == batch_size:
                batch_output += minibatch_output
                break

    return batch_output