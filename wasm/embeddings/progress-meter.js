/**
 * @file <progress-meter> custom element. Prepends a spinning icon to a <progress> while a task
 * is running. Servo does not render <progress> well, so the icon is additional visual feedback
 * that work is in progress.
 *
 * <progress-meter><progress max="100" value="0"></progress></progress-meter>
 *
 * Call start() when a task begins and stop() when it ends (or set the "active" attribute).
 * If the child <progress> has a value, the icon shows while 0 < value < max.
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
	if (!this.querySelector(":scope > .progress-meter-icon")) {
	    const template = document.createElement("template");
	    template.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-clockwise progress-meter-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false" hidden>
  <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2z"/>
  <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466"/>
</svg>`;
	    this.prepend(template.content.firstElementChild);
	}

	// Setting progress.value reflects to the attribute, so this sees every update.
	this.#observer?.disconnect();
	this.#observer = new MutationObserver(() => this.#track());
	this.#observer.observe(this, { attributes: true, attributeFilter: ["value", "max"], subtree: true });

	this.#sync();
    }

    disconnectedCallback() {
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
	this.active = progress.value > 0 && progress.value < max;
    }

    #sync() {
	const icon = this.querySelector(":scope > .progress-meter-icon");
	if (icon) {
	    // SVG elements have no hidden property, so toggle the attribute.
	    icon.toggleAttribute("hidden", !this.active);
	}
    }
}

customElements.define("progress-meter", ProgressMeter);
