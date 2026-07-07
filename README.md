# Fake-News-Detector
Multimodal Fussion Network to detect fake news, taking into account text, images and social data, utilizing rationales given from an LLM

# Branches
Branches are created based on every core part of the project, we currently have 3 main branches all coming out of main: TextDetection, ImageDetection and SocialDetection.

Every one of these branches has a `-Working` sub branch to write non tested code, so the final topology is as follows:
```text
Main
├── TextDetection
│   └── TextDetection-Working
├── ImageDetection
│   └── ImageDetection-Working
└── SocialDetection
    └── SocialDetection-Working
```

## Basic Github Branch Commands
#### Creating a Branch
```bash
git checkout <BranchName> #Name of the already existing branch out of which a new branch is going to be created
git pull 
git checkout -b <BranchName> #Name of the new branch to be created
git push -u origin <BranchName> #To publish the newly created branch
```

#### Updating a branch
This is done depending if its a working or main sub branch from main.

If its a sub branch coming straight from main, you use:
```bash
git checkout <BranchName>
git fetch origin
git merge origin/main
git push
```

And if it is a working or any other type of sub branch, you use:
```bash
git checkout <BranchName> #Branch to be updated
git merge <BranchName> #Branch straight up from that one
git push
```
