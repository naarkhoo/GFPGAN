
gunicorn 'api:create_app()' \
	 --workers 2 \
	--timeout 150 \
	--bind :5000 \
	--log-level debug \
	--threads=2
	#--worker-class gevent
	#--worker-class tornado
