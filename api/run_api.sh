
gunicorn 'api:create_app()' \
	 --workers 1 \
	--timeout 150 \
	--bind :5000 \
	--log-level debug \
	--threads=1
	#--worker-class gevent
	#--worker-class tornado
