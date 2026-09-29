# SYAUDIO data sources and licenses

SYAUDIO is a composite benchmark. The repository's MIT software license does **not** replace the licenses of its third-party datasets.

| Component | Upstream source | License | Release contents |
|---|---|---|---|
| Audio Perception (1,000) | [MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) | CC BY-NC 4.0 | Original annotations and audio, original IDs and metadata retained |
| Audio Reasoning (1,000) | [MMAR](https://huggingface.co/datasets/BoJack/MMAR) | CC BY-NC 4.0 | Original annotations and audio, including source URLs/timestamps |
| Audio Math (1,319) | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT | GSM8K test questions, SYAUDIO MCQ conversion and generated speech |
| Audio Ethics (1,000) | [MMLU](https://huggingface.co/datasets/cais/mmlu) | MIT | Moral scenarios/questions subset and generated speech |
| Human validation (300 recordings) | Collaborator contributions in [ALM-Sychophancy](https://github.com/YokeYao/ALM-Sychophancy) | Project MIT license; underlying GSM8K MIT | Annie, Danielle and Junchi readings of the same first 100 GSM8K MCQ items |

MMAU is by S Sakshi and collaborators; MMAR is by Ziyang Ma and collaborators; GSM8K is by Karl Cobbe and collaborators; MMLU is by Dan Hendrycks and collaborators. Consult the linked source cards for complete author lists, citations and source-specific notices. Upstream dataset cards were checked on 2026-09-29 and are retained in `licenses/` in the Hugging Face release.

The combined collection includes non-commercial components: comply with [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for MMAU and MMAR, retain attribution and source notices, and indicate modifications when sharing derivatives. Moving data to Hugging Face does not change these requirements. Individual original media may carry additional source notices; original provenance fields are retained.

Core audio bytes are preserved without transcoding. Archive compression changes only the container, not the underlying audio. Supplemental human readings are not additional independent benchmark questions and must not be added to the 4,319 core-question count. No training data, trained adapters, or model weights are included in this release.
