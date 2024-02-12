# MUCSUM

This directory contains the train, dev, and test splits for the official MUCSUM dataset in JSON format. The file for each split contains a JSON dictionary whose keys are document IDs (where these values match the original document IDs for the MUC-4 dataset) and whose values are lists of the MUCSUM examples associated with that document. Each example is its own JSON dictionary with the following fields:

- `instance_id`: A unique identifier for the example. We use the format `<document_id>.<number>`, where `<number>` corresponds to the `MESSAGE: TEMPLATE` field in the original MUC-4 data.
- `document`: The sentence-split document text. We use SpaCy (3.7.2) to obtain sentence boundaries from the original MUC-4 texts, which are uncased (i.e. we did not deliberately de-case them).
- `template`: The event template for this example.
- `summary`: The sentence-split summary of the event represented in the `template`, contextualized with respect to the `document`.

Each `template` contains the following fields, all of which correspond to one of the original slots/roles from the MUC-4 templates (original named given in parentheses):

- `type`: The event type; one of `attack`, `arson`, `bombing`, `forced work stoppage`, `kidnapping`, `robbery`. (`INCIDENT: TYPE`)
- `completion`: The irrealis status of the event; one of `accomplished`, `threatened`, `attempted`. (`INCIDENT: STAGE OF EXECUTION`)
- `date`: The date when the incident occurred &mdash; only if explicitly stated in the text. (Roughly corresponding to `INCIDENT: DATE`, but see caveats below)
- `location`: The location where the incident occurred &mdash; only if explicitly stated in the text. (Roughly corresponding to `INCIDENT: LOCATION`, but see caveats below)
- `perpind`: Individuals responsible for the incident. (`PERP: INDIVIDUAL ID`)
- `perporg`: Organizations responsible for the incident. (`PERP: ORGANIZATION ID`)
- `target`: Physical objects and infrastructure targeted in the incident. (`PHYS TGT: ID`)
- `victim`: Persons targeted in the incident. (`HUM TGT: DESCRIPTION`)
- `weapon`: Weapons or devices used by the perpetrators in the incident. (`INCIDENT: INSTRUMENT ID`)

A note on the `date` and `location` fields: As used in the original MUC-4 data, these are more like metadata &mdash; often reflecting outside knowledge or context about the event described that cannot actually be inferred from reading the document. In developing MUCSUM, we wanted it to be possible to generate the summaries *without* such additional knowledge &mdash; relying only on the text. As such, our `date` and `location` fields annotate dates and locations only when the date and location of the incident is explicitly mentioned.

A note on `perpind`, `perporg`, `target`, `victim`, `weapon`: These are the five "string-fill" slots of the MUC-4 templates &mdash; the subset of the original set of 24 slots that researchers working with MUC-4 typically evaluate their systems against. "String-fill" just means that these slots take entities as values (as opposed to categorical values). The annotations for these slots in the original data is rather unconventional by modern standards in two respects:
- Only partial coreference information is provided for each entity. This will generally include any proper names (e.g. "FMLN") and nominal expressions (e.g. "terrorists"), but will always exclude pronouns.
- Textual offsets of the mentions are not provided.

In cases where the MUC-4 data provides multiple mentions for a particular entity, we have generally used just *one* of these in the summary (typically, the most informative), and we use this single mention to represent the associated entity in the appropriate slot in the `template`. Thus, the values of the string-fill slots in each `template` are lists of entity mentions, each of which refers to a distinct entity. We may add back the partial coreference information to these files in the future, but for now, this information can be found in the preprocessed MUC-4 files here: https://github.com/xinyadu/gtt/tree/master/data/muc/processed.
