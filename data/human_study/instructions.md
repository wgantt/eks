# Overview

In this task, you are given a set of 30 documents, along with:

1. Some information about an event described in each document in the form of an "event template."
2. A set of six candidate summaries of that event, based on the document.

Importantly, these summaries are **not** meant to be summaries of the document as a whole, but rather of the **specific** event that is presented along with the document. For this task, we are asking you to evaluate the quality of each summary on a scale of 1 (very poor) to 5 (excellent). Note that we are not necessarily asking for a total ranking here: each summary is to be evaluated independently on this scale, so you may give multiple summaries the same score. We provide more details on how to evaluate summaries in the [Scoring Summaries](#scoring-summaries) section further down. But first, we'll talk about the events on which these summaries are based.

# Events

The events you will see in this task will be of one of four types, all of which are generally acts of terrorism:

- `robbery`: incidents of robbery or theft
- `bombing`: bombings
- `forced work stoppage`: incidents of transportation systems or other networks being forcibly stopped
- `attack`: a catch-all event type that includes kinds of attacks other than bombings (e.g. assassinations)

While each document may describe multiple events, we will only ask you about a single event for each document. The event is represented by an event *template*, which contains the following information (in addition to the `type`, mentioned above):

- `individual perpetrators`: individual people or groups of people responsible for the incident.
- `organizations responsible`: the organization(s) that the perpetrators are affiliated with or that are mentioned as being responsible for the attack/bombing/robbery/forced work stoppage.
- `victims`: people harmed or killed in the event.
- `physical targets`: cars, buildings, or other physical infrastructure that is damaged or destroyed in the event.
- `weapons`: weapons used by the perpetrators during the event.
- `stage of completion`: whether the event actually happened (`accomplished`), was `attempted` but not fully completed, or was merely `threatened`.
- `date`: when the event took place (or will take place).
- `location`: where the event took place (or will take place).

For each of these fields, there may be multiple values (e.g. multiple `weapon`s or multiple `individual perpetrators`), or there may be none, depending on what information is provided about the event in the document. In judging a summary, you should base your score primarily on how well it captures the information provided in the event template. More on this below.

# Scoring Summaries

As we say above, the summaries are meant to be about one *specific event*, so your score for a summary should reflect how good it is *as a summary of that event only*. More specifically, your score for a given summary should be based on the following attributes (in descending order of importance):

- **Consistency**/**Factuality**: *Does the summary make only true statements about the event in question, given what the document says about that event?*
  - Summaries that make factual errors of any kind should be penalized.
- **Adequacy**: *Does the summary adequately capture all of the information contained in the event template?*
  - Summaries that omit details about any of the participants in the event (the perpetrators, victims, date, location, etc.) should be penalized.
  - One partial exception to this is the `location` field. Sometimes, an event template will have multiple values in the `location` field; it is okay if the summary mentions only one of these locations, but it should be the most specific one. (E.g. if both `California` and `San Francisco` are included in the `location` field, the summary doesn't have to mention both, but if it mentions one, it should ideally mention `San Francisco`. Otherwise, the summary should be penalized.)
- **Coherence**: *Does the summary make sense on its own, as a standalone description of the event?*
  - Summaries that require you to go read the document in order to understand what they mean (or that don't make sense even then) should be penalized.
- **Relevancy**: *Does the summary include only information that is relevant to the event in question*?
  - Summaries that include irrelevant or superfluous information, or information about some event other than the one represented by the event template, should be penalized.
- **Fluency**: *Does the summary sound reasonable natural (like something a native English speaker might actually write)?*
  - Summaries that are disfluent or that sound unnatural should be penalized.

Oftentimes, some of the summaries may be very similar to each other. It is totally fine to give multiple summaries the same score if you think they are of comparable quality!

You should enter your score for each summary in the `score` field. The default value for each summary is 3. Please do not use half scores (1.5, 2.5, etc).

As a final note, note that the text of both the documents and summaries is completely lowercased (even words at the beginning of sentences, names, etc.). This is a limitation of the data itself and we apologize if it hinders readability. Please do your best to accommodate this.

When you are done, please return the file to me, saving it as `<your_name>_data.json`.