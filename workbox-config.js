module.exports = {
	globDirectory: './',
	globPatterns: [
		'**/*.{css,html,js,jpg,png,svg,json,scss}'
	],
	swDest: 'service-worker.js',
	ignoreURLParametersMatching: [
		/^utm_/,
		/^fbclid$/
	]
};