# LLM-Driven Knowledge Graph Construction for Human Smuggling Networks

Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer valuable insights but are unstructured, lexically dense, and filled with ambiguous or shifting references—posing challenges for automated knowledge graph (KG) construction. Existing KG methods often rely on static templates and lack coreference resolution, while recent LLM-based approaches frequently produce noisy, fragmented graphs due to hallucinations and duplicate nodes caused by a lack of guided extraction. We propose CORE-KG, a modular framework for building interpretable KGs from legal texts. It uses a two-step pipeline: (1) type-aware coreference resolution via sequential, structured LLM prompts, and (2) entity and relationship extraction using domain-guided instructions, built on an adapted GraphRAG framework. CORE-KG reduces node duplication by 33.28% and legal noise by 38.37% compared to a GraphRAG-based baseline—resulting in cleaner and more coherent graph structures. These improvements make CORE-KG a strong foundation for analyzing complex criminal networks.

## Instructions to Run the Code

The complete code for both the CORE-KG pipeline and the baseline is provided. To run the pipeline, you need to have Ollama installed locally. You can find the installation instructions here: https://ollama.com/download

Make sure you are using Python 3.12.

To run the baseline, you will need GraphRAG version 0.3.2.


## Commands

All inputs are taken through the terminal.

### 1. Run Coreference Resolution (CORE-KG)

The primary file for running the CORE-KG coreference resolution module is `resolve_coref_pipeline.py`.

Use the following template to run the resolution pipeline. Fill in the paths to your input file, output folder, and prompt files as needed:

```bash
python resolve_coref_pipeline.py \
  --input-file <input_txt_file> \
  --output-folder <output_folder> \
  --person-prompt <path_to_person_prompt> \
  --routes-prompt <path_to_routes_prompt> \
  --location-prompt <path_to_location_prompt> \
  --mot-prompt <path_to_transportation_prompt> \
  --moc-prompt <path_to_communication_prompt> \
  --organization-prompt <path_to_organization_prompt> \
  --smuggleditems-prompt <path_to_smuggleditems_prompt> \
  --model <model_name_in_ollama>
```

#### CORE-KG Prompts
##### Coreference Resolution Prompts
All the required prompts for coreference resolution are present in the `core-kg/coreference-resolution/prompts/` folder.

##### KG Construction Prompts
All the required prompts for KG construction are present in the `core-kg/kgconstruction/ragtest/prompts/` folder.

### 2. Run KG Construction and Baseline (GraphRAG)

The primary file for running the KG construction module and the baseline is `index.py`. Use the command below:

```bash
python index.py --root ./ragtest
```

Make sure you are in the correct directory. Follow the full setup and usage instructions provided here:  
[https://github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)

#### Baseline (GraphRAG) Prompts
All the required prompts for the baseline model are present in the `baseline/ragtest/prompts/` folder.


## Conclusion

CORE-KG demonstrates how structured prompt engineering and type-aware coreference resolution can significantly enhance the quality of LLM-generated knowledge graphs. By reducing redundancy, legal noise, and misclassification, CORE-KG produces more coherent and actionable representations of smuggling networks, offering a strong foundation for further analysis in the legal and policy domains.
