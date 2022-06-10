/* ---- tsParticles config ---- */

tsParticles.load("tsparticles", {
  fpsLimit: 60,
  particles: {
    number: {
      value: 70,
      density: {
        enable: true,
        value_area: 900
      }
    },
    color: {
      value: ["#ccccccff"]   /*#ccccccff*/
    },
    shape: {
      type: ["circle"],
      stroke: {
        width: 0,
        color: "#fff"
      },
      polygon: {
        nb_sides: 5
      }
    },
    opacity: {
      value: 0.8,
      random: false,
      anim: {
        enable: false,
        speed: 1,
        opacity_min: 0.1,
        sync: false
      }
    },
    size: {
      value: 6,
      random: true,
      anim: {
        enable: false,
        speed: 8,
        size_min: 10,
        sync: false
      }
    },
    line_linked: {
      enable: true,
      distance: 150,
      color: "#808080",
      opacity: 0.2,
      width: 1.5
    },
    move: {
      enable: true,
      speed: 1,
      direction: "none",
      random: false,
      straight: false,
      out_mode: "stubs",
      bounce: false,
      attract: {
        enable: false,
        rotateX: 600,
        rotateY: 1200
      }
    }
  },
  interactivity: {
    detect_on: "canvas",
    events: {
      onhover: {
        enable: true,
        mode: "grab"
      },
      onclick: {
        enable: true,
        mode: "push"
      },
      resize: true
    },
    modes: {
      grab: {
        distance: 140,
        line_linked: {
          opacity: 1
        }
      },
      bubble: {
        distance: 400,
        size: 40,
        duration: 2,
        opacity: 8,
        speed: 3
      },
      repulse: {
        distance: 200,
        duration: 0.4
      },
      push: {
        particles_nb: 4
      },
      remove: {
        particles_nb: 2
      }
    }
  },
  retina_detect: true
});
