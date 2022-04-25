module.exports = {
	globDirectory: './',
	globPatterns: [
		'**/*.{css,html,js,jpg,png,svg,json,scss}'
	],
	swDest: 'sw.js',
	ignoreURLParametersMatching: [
		/^utm_/,
		/^fbclid$/
	]
};