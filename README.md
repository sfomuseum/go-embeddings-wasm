# go-embeddings-wasm

This is the source code for the `llama.cpp` in WebAssembly (WASM) demo application to generate and query vector embeddings. For background, please read the [Shared vector embeddings updates: Support for EmbeddingGemma2 and llama.cpp](https://millsfield.sfomuseum.org/blog/2026/11/23/llamacpp/) blog post.

## Getting started

Please consult the documentation in the [wasm](wasm) folder.

## Important

_The following is extracted from the blog post, mentioned above:_

There is an important caveat to note about this demo application: It was largely written with the assistance of machines (large language models). Obviously, I was comfortable enough to use these systems for the time it took to build the application, but that's about where it stops. Why, then, did I let a machine write most of this code? Because I could. The opportunity [to use frontier models](https://www.aaronland.info/weblog/2026/09/17/overachievers/#ff26-004) presented itself and I decided to see, for myself, what the fuss was all about. In a nutshell: It's pretty easy to understand why people who write software [are excited or filled with dread](https://www.tbray.org/ongoing/When/202x/2026/11/18/AI-Patterns) about these tools, often at the same time. If the measure is simply time – efficiency – then there's no question that machine-assistance can be a benefit. The simple fact that time and efficiency are the principal measures we fall back on, though, tells us something about our priorities, indepedent of machine-assisted software engineering.

The application did not self-actualize. It required a prolonged back-and-forth to address errors and misunderstandings. It was still faster than the time it would have taken me to decipher the many layers of code, often lacking approachable documentation, required to run `llama.cpp` inside WebAssembly from a complex web application involving [ServiceWorkers](https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorker). That's not nothing. It would be easier to pretend that it is (nothing), but it's not.

Is it enough, though, to offset [all the "externalities" that frontier models, and the companies who run them, impose](https://www.tbray.org/ongoing/When/202x/2026/10/19/The-AI-Temptation) on everything else? I am not sure it is. It is true that, both individually and collectively, we have a long history of being able to ignore externalities when they offer convenience and the cost of that convenience is born by someone else, but that is, at best, an unsatisfying rationalization.

Despite that, I _did_ use machine-assistance to create this application. I have also read [every single line of code it produced](https://github.com/sfomuseum/go-embeddings-wasm), tweaking it along the way where necessary. In its own way this process afforded the opportunity to learn, and begin internalizing, subjects that were new to me; things that time and circumstance would have otherwise argued against. Again, this is not nothing which lends itself to the otherwise reasonable question: Is there a way to use these tools responsibly, in moderation? As with the "externalities", this is not clear to me that there is. So, for the time being, I am reluctant to do it again.

## See also

* https://github.com/hybridgroup/yzma
* https://github.com/sfomuseum/go-embeddings