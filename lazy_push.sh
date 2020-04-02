#!/bin/bash
if [ "$(git config --global user.name)" != "xiaohangzhan" ]; then
    echo "wrong git user"
fi
git add --all
git commit -m "update"
git pull
git push origin master
