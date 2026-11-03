# Agent skills for RadVel

Skills that teach AI coding agents (Claude Code and other tools that read `SKILL.md`) to use RadVel correctly.

- `writing-radvel-setup-files`: what a setup file must contain, choosing a basis and priors, instrument names, time systems.
- `fitting-rvs-with-radvel`: the `radvel` CLI pipeline, flags that trip agents, MCMC, outputs, running in Docker, checks before quoting numbers.

Both were checked against radvel 1.6.7 by running agents on bundled example data with and without the skills.

## Install

From the Orrery registry (https://orrery.science/install), or copy the folders into your skills directory:

```bash
cp -r skills/fitting-rvs-with-radvel skills/writing-radvel-setup-files ~/.claude/skills/
```
