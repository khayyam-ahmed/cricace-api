web: chmod +x /prereq.sh && /prereq.sh
web: gunicorn -w  4 -k uvicorn.workers.UvicornWorker main:app