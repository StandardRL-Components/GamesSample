import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where a player controls a 'neutron' to navigate through a
    pulsating 'electron' field. The goal is to survive for a set duration while
    'synchronizing' with a number of electron pulses.

    **Visuals:**
    - A minimalist, high-contrast sci-fi aesthetic.
    - Player (Neutron): A bright blue circle with a glowing aura.
    - Obstacles (Electrons): White circles that pulse, turning bright red and expanding
      when they become dangerous.
    - Background: A deep black void.
    - Effects: Particle bursts provide satisfying feedback for successful actions.

    **Gameplay:**
    - The player moves the neutron horizontally.
    - Electrons pulse in a rhythmic, desynchronized cycle.
    - The player must avoid touching electrons during their active (red) pulse phase.
    - Rewards are given for survival, and a larger bonus is given for being inside an
      electron's pulse radius at the exact moment its pulse cycle completes, signifying
      a 'synchronization' or 'pass'.

    **State & Goal:**
    - State: Neutron position, electron pulse timers, game timer, pulses passed.
    - Victory: Survive for 750 steps (45 seconds) AND achieve 10 pulse passes.
    - Failure: Collide with an active electron.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a neutron through a pulsating electron field, avoiding active pulses "
        "while synchronizing with them to score points."
    )
    user_guide = "Use ← and → arrow keys to move the neutron left and right."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Pygame rendering FPS
    GAME_LOGIC_STEPS_PER_SEC = 15

    # Colors
    COLOR_BG = (10, 5, 15)
    COLOR_NEUTRON = (0, 191, 255)
    COLOR_NEUTRON_GLOW = (0, 100, 150)
    COLOR_ELECTRON_INACTIVE = (220, 220, 255)
    COLOR_ELECTRON_ACTIVE = (255, 50, 50)
    COLOR_TEXT = (255, 255, 255)

    # Game Parameters
    MAX_STEPS = 750 # 45 seconds at 15 steps/sec (approx)
    PULSE_GOAL = 10
    NUM_ELECTRONS = 15
    PULSE_CYCLE_SECONDS = 2.0
    PULSE_ACTIVE_DURATION_SECONDS = 0.4 # The dangerous part of the pulse

    # Neutron Physics
    NEUTRON_RADIUS = 10
    NEUTRON_ACCEL = 1.0
    NEUTRON_DRAG = 0.85

    # Electron Physics
    ELECTRON_BASE_RADIUS = 15
    ELECTRON_PULSE_RADIUS_MAX = 45

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 50)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.neutron_pos = np.array([0.0, 0.0])
        self.neutron_vel = np.array([0.0, 0.0])
        self.electrons = []
        self.particles = []
        self.pulses_passed = 0
        self.passed_this_cycle = set()

        # Derived constants
        self.pulse_cycle_steps = int(self.PULSE_CYCLE_SECONDS * self.GAME_LOGIC_STEPS_PER_SEC)
        self.pulse_active_start_step = self.pulse_cycle_steps - int(self.PULSE_ACTIVE_DURATION_SECONDS * self.GAME_LOGIC_STEPS_PER_SEC)

        # Run validation
        # self.validate_implementation() # This is non-standard and can be commented out.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pulses_passed = 0

        self.neutron_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.neutron_vel = np.array([0.0, 0.0], dtype=float)

        self.electrons = []
        start_pos = self.neutron_pos.copy()
        safe_spawn_radius = self.ELECTRON_PULSE_RADIUS_MAX + self.NEUTRON_RADIUS + 5 # Buffer to prevent spawn kills

        for i in range(self.NUM_ELECTRONS):
            # Ensure electrons don't spawn on top of the player's start position
            while True:
                pos = np.array([
                    self.np_random.uniform(self.ELECTRON_PULSE_RADIUS_MAX, self.SCREEN_WIDTH - self.ELECTRON_PULSE_RADIUS_MAX),
                    self.np_random.uniform(self.ELECTRON_PULSE_RADIUS_MAX, self.SCREEN_HEIGHT - self.ELECTRON_PULSE_RADIUS_MAX)
                ])
                if np.linalg.norm(pos - start_pos) > safe_spawn_radius:
                    break
            self.electrons.append({
                'pos': pos,
                'pulse_timer': self.np_random.integers(0, self.pulse_cycle_steps),
                'id': i
            })

        self.particles = []
        self.passed_this_cycle = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        self._handle_input(action)
        self._update_neutron()

        collision, pass_reward = self._update_electrons()
        reward += pass_reward
        self.score += pass_reward

        self._update_particles()

        self.steps += 1

        # Survival reward
        reward += 0.1
        self.score += 0.1

        terminated = self._check_termination(collision)
        truncated = False # This environment does not truncate based on time limit

        if terminated:
            if collision:
                reward = -100.0
            elif self.pulses_passed >= self.PULSE_GOAL:
                reward = 100.0
            self.score += reward # Add terminal reward to final score
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        if movement == 3:  # Left
            self.neutron_vel[0] -= self.NEUTRON_ACCEL
        elif movement == 4:  # Right
            self.neutron_vel[0] += self.NEUTRON_ACCEL
        # Actions 0, 1, 2 are no-ops for movement

    def _update_neutron(self):
        self.neutron_pos += self.neutron_vel
        self.neutron_vel *= self.NEUTRON_DRAG

        # Clamp position to screen bounds
        self.neutron_pos[0] = np.clip(
            self.neutron_pos[0], self.NEUTRON_RADIUS, self.SCREEN_WIDTH - self.NEUTRON_RADIUS
        )
        self.neutron_pos[1] = self.SCREEN_HEIGHT / 2 # Lock to vertical center

    def _update_electrons(self):
        collision = False
        pass_reward = 0.0

        for electron in self.electrons:
            electron['pulse_timer'] = (electron['pulse_timer'] + 1)

            # --- Pulse Pass Check ---
            if electron['pulse_timer'] == self.pulse_cycle_steps:
                dist = np.linalg.norm(self.neutron_pos - electron['pos'])
                if dist <= self.ELECTRON_PULSE_RADIUS_MAX and electron['id'] not in self.passed_this_cycle:
                    self.pulses_passed += 1
                    pass_reward += 1.0
                    self.passed_this_cycle.add(electron['id'])
                    self._create_particles(self.neutron_pos, self.COLOR_ELECTRON_INACTIVE, 20)

            # Reset timer and cycle tracker
            if electron['pulse_timer'] >= self.pulse_cycle_steps:
                electron['pulse_timer'] = 0
                if electron['id'] in self.passed_this_cycle:
                    self.passed_this_cycle.remove(electron['id'])

            # --- Collision Check ---
            is_active, current_radius = self._get_pulse_state(electron['pulse_timer'])
            if is_active:
                dist = np.linalg.norm(self.neutron_pos - electron['pos'])
                if dist < current_radius + self.NEUTRON_RADIUS:
                    collision = True
                    self._create_particles(self.neutron_pos, self.COLOR_ELECTRON_ACTIVE, 50, 5)

        return collision, pass_reward

    def _get_pulse_state(self, timer):
        is_active = timer >= self.pulse_active_start_step

        # Smooth sine wave for radius expansion
        progress = timer / self.pulse_cycle_steps
        pulse_factor = abs(math.sin(progress * math.pi))

        current_radius = self.ELECTRON_BASE_RADIUS + (self.ELECTRON_PULSE_RADIUS_MAX - self.ELECTRON_BASE_RADIUS) * pulse_factor

        return is_active, current_radius

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, color, count, speed_mult=2.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _check_termination(self, collision):
        if collision:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = p['color']
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*color, alpha)
            )

        # Render electrons
        for electron in self.electrons:
            is_active, radius = self._get_pulse_state(electron['pulse_timer'])
            color = self.COLOR_ELECTRON_ACTIVE if is_active else self.COLOR_ELECTRON_INACTIVE

            # Antialiased filled circle
            pygame.gfxdraw.filled_circle(self.screen, int(electron['pos'][0]), int(electron['pos'][1]), int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, int(electron['pos'][0]), int(electron['pos'][1]), int(radius), color)

        # Render neutron with glow
        glow_radius = int(self.NEUTRON_RADIUS * 1.8)
        glow_alpha = 90
        pygame.gfxdraw.filled_circle(self.screen, int(self.neutron_pos[0]), int(self.neutron_pos[1]), glow_radius, (*self.COLOR_NEUTRON_GLOW, glow_alpha))

        pygame.gfxdraw.filled_circle(self.screen, int(self.neutron_pos[0]), int(self.neutron_pos[1]), self.NEUTRON_RADIUS, self.COLOR_NEUTRON)
        pygame.gfxdraw.aacircle(self.screen, int(self.neutron_pos[0]), int(self.neutron_pos[1]), self.NEUTRON_RADIUS, self.COLOR_NEUTRON)

    def _render_ui(self):
        # Pulses passed
        pulse_text = self.font.render(f"PULSES: {self.pulses_passed} / {self.PULSE_GOAL}", True, self.COLOR_TEXT)
        self.screen.blit(pulse_text, (10, 10))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,180))

        if self.pulses_passed >= self.PULSE_GOAL and self.steps >= self.MAX_STEPS:
            text = "SYNCHRONIZATION COMPLETE"
            color = self.COLOR_NEUTRON
        else:
            text = "DECOHERENCE"
            color = self.COLOR_ELECTRON_ACTIVE

        game_over_text = self.font_large.render(text, True, color)
        text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(s, (0,0))
        self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pulses_passed": self.pulses_passed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)


if __name__ == "__main__":
    # --- Example of how to run the environment ---
    env = GameEnv()

    # --- Manual Play ---
    # Use Left/Right arrow keys to control the neutron
    obs, info = env.reset()
    done = False

    # Pygame window for human interaction
    # To run with a display, comment out the os.environ line at the top
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Neutron Pulse")
        clock = pygame.time.Clock()
        real_display = True
    except pygame.error:
        print("Pygame display could not be initialized (running in headless mode).")
        real_display = False


    total_reward = 0

    while not done:
        action = [0, 0, 0] # Default no-op
        if real_display:
            # Map keyboard input to action space
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            action = [movement, 0, 0] # Space and Shift are not used

            # Handle window closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        if done:
            break

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if real_display:
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Slow down the loop to match the game's intended logic speed for playability
            clock.tick(env.GAME_LOGIC_STEPS_PER_SEC)

    print(f"Episode finished. Final Score: {info['score']:.2f}, Pulses Passed: {info['pulses_passed']}")

    env.close()