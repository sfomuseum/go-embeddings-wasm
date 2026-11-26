/**
 * @file <progress-meter> custom element. Prepends a spinning icon to a <progress> while a task
 * is running. Servo does not render <progress> well, so the icon is additional visual feedback
 * that work is in progress.
 *
 * <progress-meter><progress max="100" value="0"></progress></progress-meter>
 *
 * Call start() when a task begins and stop() when it ends (or set the "active" attribute).
 * If the child <progress> has a value, the icon shows once 0 < value < max and hides at max.
 *
 * The element can be connected before the parser has read its children, so nothing here
 * depends on the children being present in connectedCallback.
 */
class ProgressMeter extends HTMLElement {

    #observer = null;

    static get observedAttributes() {
	return ["active"];
    }

    connectedCallback() {
	if (!this.#icon()) {
	    const ns = "http://www.w3.org/2000/svg";
	    const svg = document.createElementNS(ns, "svg");
	    svg.setAttribute("width", "16");
	    svg.setAttribute("height", "16");
	    svg.setAttribute("viewBox", "0 0 16 16");
	    svg.setAttribute("fill", "#333");
	    svg.setAttribute("aria-hidden", "true");
	    svg.setAttribute("focusable", "false");
	    for (const [d, rule] of [
		["M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2z", "evenodd"],
		["M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466", null],
	    ]) {
		const path = document.createElementNS(ns, "path");
		path.setAttribute("d", d);
		if (rule) {
		    path.setAttribute("fill-rule", rule);
		}
		svg.append(path);
	    }
	    // The span is what spins and hides; SVG roots handle transforms and [hidden] inconsistently.
	    const icon = document.createElement("span");
	    icon.className = "progress-meter-icon";
	    icon.hidden = true;
	    icon.append(svg);
	    this.prepend(icon);
	}

	// Setting progress.value reflects to the attribute, so this sees every update.
	this.#observer?.disconnect();
	this.#observer = new MutationObserver(() => this.#track());
	this.#observer.observe(this, { attributes: true, attributeFilter: ["value", "max"], subtree: true });

	this.#sync();
    }

    disconnectedCallback() {
	clearInterval(this.#timer);
	this.#timer = 0;
	this.#observer?.disconnect();
	this.#observer = null;
    }

    attributeChangedCallback() {
	this.#sync();
    }

    /** Whether the icon is shown and spinning. */
    get active() {
	return this.hasAttribute("active");
    }

    set active(on) {
	this.toggleAttribute("active", !!on);
    }

    /** Shows the icon and starts it rotating. */
    start() {
	this.active = true;
    }

    /** Hides the icon. */
    stop() {
	this.active = false;
    }

    /** Shows the icon while progress is partway; hides it at 0 or when complete. */
    #track() {
	const progress = this.querySelector("progress");
	if (!progress || !progress.hasAttribute("value")) {
	    return;
	}
	const max = progress.max > 0 ? progress.max : 1;
	// A value of 0 leaves the state alone so start() before the first update is kept.
		if (progress.value >= max) {
		    this.active = false;
		} else if (progress.value > 0) {
		    this.active = true;
		}
    }

    /** The icon among the direct children (avoids :scope, which may be unsupported). */
    #icon() {
		for (const child of this.children) {
		    if (child.classList.contains("progress-meter-icon")) {
			return child;
		    }
		}
		return null;
    }

    #timer = 0;

    #sync() {
	const icon = this.#icon();
	if (icon) {
	    icon.hidden = !this.active;
	    // Rotate from script: Servo does not advance CSS animations while it is busy,
	    // and a timer keeps the icon turning whenever the page thread runs at all.
	    if (this.active && !this.#timer) {
		const t0 = performance.now();
		this.#timer = setInterval(() => {
		    const deg = ((performance.now() - t0) / 1000 * 360) % 360;
		    icon.style.transform = `rotate(${deg}deg)`;
		}, 40);
	    } else if (!this.active && this.#timer) {
		clearInterval(this.#timer);
		this.#timer = 0;
	    }
	}
    }
}

customElements.define("progress-meter", ProgressMeter);
