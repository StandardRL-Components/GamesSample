import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:21:47.890301
# Source Brief: brief_00359.md
# Brief Index: 359
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Gravity Well Collector'.

    The player manipulates three synchronized gravity wells (Red, Green, Blue)
    to attract and collect 50 asteroids. The game is set in a circular arena
    that continuously shrinks, creating time pressure. Collecting an asteroid
    slightly weakens the corresponding well's attraction power.

    The goal is to collect all asteroids before the arena shrinks to a
    critical size.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three gravity wells to attract and collect floating asteroids within a shrinking arena. "
        "Collect all asteroids before the arena collapses to win."
    )
    user_guide = (
        "Controls: Use ↑, ↓, ← to activate the Red, Green, and Blue wells. "
        "Use →, space, and shift to deactivate them."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

    COLOR_BG = (10, 15, 30)
    COLOR_ARENA = (200, 200, 255)
    COLOR_ASTEROID = (255, 255, 255)
    COLOR_TEXT = (220, 220, 255)
    WELL_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]

    INITIAL_ARENA_RADIUS = 180
    MIN_ARENA_RADIUS_FACTOR = 0.2
    ARENA_SHRINK_RATE = 0.001  # 0.1% of initial radius per step

    TOTAL_ASTEROIDS = 50
    ASTEROID_RADIUS = 2
    ASTEROID_SPEED_SCALAR = 1.0

    WELL_RADIUS = 12
    INITIAL_WELL_STRENGTH = 1.0
    WELL_STRENGTH_DECAY = 0.95  # 5% reduction per collection

    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_ui = pygame.font.SysFont("Consolas", 14)

        # Game state variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_arena_radius = 0
        self.asteroids = []
        self.wells = []
        self.particles = []


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_arena_radius = self.INITIAL_ARENA_RADIUS
        self.particles = []

        # Initialize Wells
        self.wells = []
        angle_step = 360 / 3
        for i in range(3):
            angle = math.radians(90 + i * angle_step)
            well_pos = pygame.Vector2(
                self.CENTER[0] + math.cos(angle) * self.INITIAL_ARENA_RADIUS * 0.6,
                self.CENTER[1] + math.sin(angle) * self.INITIAL_ARENA_RADIUS * 0.6
            )
            self.wells.append({
                "pos": well_pos,
                "color": self.WELL_COLORS[i],
                "strength": self.INITIAL_WELL_STRENGTH,
                "active": False,
            })

        # Initialize Asteroids
        self.asteroids = []
        while len(self.asteroids) < self.TOTAL_ASTEROIDS:
            angle = self.np_random.uniform(0, 2 * math.pi)
            radius = self.np_random.uniform(0, self.INITIAL_ARENA_RADIUS * 0.95)
            pos = pygame.Vector2(
                self.CENTER[0] + math.cos(angle) * radius,
                self.CENTER[1] + math.sin(angle) * radius
            )
            # Ensure asteroids don't spawn on top of wells
            if all(pos.distance_to(w['pos']) > self.WELL_RADIUS * 2 for w in self.wells):
                self.asteroids.append({"pos": pos})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_actions(action)
        reward = self._update_game_state()
        self._update_arena()
        self._update_particles()

        self.steps += 1
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        # Truncated is always False as termination is handled by game logic
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        activate_requests = [False, False, False]
        if movement == 1: activate_requests[0] = True # Up -> Activate Red
        if movement == 2: activate_requests[1] = True # Down -> Activate Green
        if movement == 3: activate_requests[2] = True # Left -> Activate Blue

        deactivate_requests = [False, False, False]
        if movement == 4: deactivate_requests[0] = True # Right -> Deactivate Red
        if space_held:    deactivate_requests[1] = True # Space -> Deactivate Green
        if shift_held:    deactivate_requests[2] = True # Shift -> Deactivate Blue

        # Deactivation takes priority over activation
        for i in range(3):
            if deactivate_requests[i]:
                self.wells[i]['active'] = False
            elif activate_requests[i]:
                self.wells[i]['active'] = True

    def _update_game_state(self):
        reward = 0
        asteroids_to_remove = []

        active_wells = [w for w in self.wells if w['active']]

        for i, asteroid in enumerate(self.asteroids):
            total_force = pygame.Vector2(0, 0)
            if active_wells:
                for well in active_wells:
                    vec_to_well = well['pos'] - asteroid['pos']
                    dist = vec_to_well.length()
                    if dist > 1: # Avoid division by zero
                        force = vec_to_well.normalize() * well['strength'] * (self.INITIAL_ARENA_RADIUS / max(dist, 1))
                        total_force += force

            asteroid['pos'] += total_force * self.ASTEROID_SPEED_SCALAR

            # Check for collection
            for well_idx, well in enumerate(self.wells):
                if asteroid['pos'].distance_to(well['pos']) < self.WELL_RADIUS:
                    asteroids_to_remove.append(i)
                    self.score += 1
                    reward += 0.1
                    well['strength'] *= self.WELL_STRENGTH_DECAY
                    self._create_particles(asteroid['pos'], well['color'])
                    # SFX: # sfx_collect.wav
                    break # Asteroid can only be collected by one well
            else: # Check for out of bounds only if not collected
                if asteroid['pos'].distance_to(pygame.Vector2(self.CENTER)) > self.current_arena_radius:
                    asteroids_to_remove.append(i)
                    # SFX: # sfx_asteroid_lost.wav

        # Remove asteroids in reverse order to avoid index errors
        for i in sorted(list(set(asteroids_to_remove)), reverse=True):
            del self.asteroids[i]

        return reward

    def _update_arena(self):
        self.current_arena_radius -= self.INITIAL_ARENA_RADIUS * self.ARENA_SHRINK_RATE
        self.current_arena_radius = max(
            self.current_arena_radius,
            self.INITIAL_ARENA_RADIUS * self.MIN_ARENA_RADIUS_FACTOR
        )

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 25 + 1)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(i)
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _check_termination(self):
        terminated = False
        reward = 0

        if self.score >= self.TOTAL_ASTEROIDS:
            terminated = True
            reward = 100.0
            # SFX: # sfx_win.wav
        elif self.current_arena_radius <= self.INITIAL_ARENA_RADIUS * self.MIN_ARENA_RADIUS_FACTOR:
            terminated = True
            reward = -100.0
            # SFX: # sfx_lose.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -100.0
            # SFX: # sfx_timeout.wav
        
        return terminated, reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Arena
        pygame.gfxdraw.aacircle(
            self.screen, int(self.CENTER[0]), int(self.CENTER[1]),
            int(self.current_arena_radius), self.COLOR_ARENA
        )

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 25))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((2,2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (1,1), 1)
            self.screen.blit(temp_surf, (int(p['pos'].x - 1), int(p['pos'].y - 1)))


        # Draw Asteroids
        for asteroid in self.asteroids:
            pygame.gfxdraw.filled_circle(
                self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y),
                self.ASTEROID_RADIUS, self.COLOR_ASTEROID
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y),
                self.ASTEROID_RADIUS, self.COLOR_ASTEROID
            )

        # Draw Wells
        for well in self.wells:
            pos_int = (int(well['pos'].x), int(well['pos'].y))
            # Base color
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.WELL_RADIUS, well['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.WELL_RADIUS, well['color'])

            if well['active']:
                # Glowing core
                glow_color = tuple(min(255, c + 80) for c in well['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.WELL_RADIUS // 2, glow_color)

                # Pulsating field
                pulse_progress = (self.steps % 30) / 30
                pulse_alpha = int(100 * (math.sin(pulse_progress * 2 * math.pi) * 0.5 + 0.5))
                pulse_radius = int(self.WELL_RADIUS * 2.5)
                
                temp_surf = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
                field_color = (*well['color'], pulse_alpha)
                pygame.gfxdraw.filled_circle(temp_surf, pulse_radius, pulse_radius, pulse_radius, field_color)
                pygame.gfxdraw.aacircle(temp_surf, pulse_radius, pulse_radius, pulse_radius, field_color)
                self.screen.blit(temp_surf, (pos_int[0] - pulse_radius, pos_int[1] - pulse_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score display
        score_text = f"ASTEROIDS: {self.score} / {self.TOTAL_ASTEROIDS}"
        text_surface = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Arena size display
        arena_percent = (self.current_arena_radius / self.INITIAL_ARENA_RADIUS) * 100
        arena_text = f"ARENA INTEGRITY: {arena_percent:.1f}%"
        text_surface = self.font_main.render(arena_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surface, text_rect)

        # Well status indicators
        well_labels = ["R", "G", "B"]
        for i, well in enumerate(self.wells):
            base_x = self.SCREEN_WIDTH - 110 + (i * 35)
            base_y = 45
            
            label_surf = self.font_ui.render(well_labels[i], True, well['color'])
            self.screen.blit(label_surf, (base_x + 6, base_y))

            status_color = self.WELL_COLORS[i] if well['active'] else (50, 50, 60)
            pygame.draw.rect(self.screen, status_color, (base_x, base_y + 18, 20, 10))
            pygame.draw.rect(self.screen, self.COLOR_ARENA, (base_x, base_y + 18, 20, 10), 1)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "arena_radius_percent": (self.current_arena_radius / self.INITIAL_ARENA_RADIUS) * 100,
            "well_strengths": [w['strength'] for w in self.wells]
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Note: The main block uses a different control scheme for human playability.
    # The environment's action space is defined in _handle_actions.
    
    # Un-dummy the video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Gravity Well Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("R Well: W (activate) / S (deactivate)")
    print("G Well: A (activate) / D (deactivate)")
    print("B Well: Q (activate) / E (deactivate)")
    print("Reset: R")
    print("----------------------\n")

    # This state needs to be maintained for human play to be intuitive
    well_states = [False, False, False]

    while running:
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            well_states = [False, False, False]

        # Convert human input to MultiDiscrete action space
        # We process key presses to toggle states, then map those states to an action
        action = [0, 0, 0] # default no-op
        keys = pygame.key.get_pressed()

        # Map key presses to well states
        if keys[pygame.K_w]: well_states[0] = True
        if keys[pygame.K_s]: well_states[0] = False
        if keys[pygame.K_a]: well_states[1] = True
        if keys[pygame.K_d]: well_states[1] = False
        if keys[pygame.K_q]: well_states[2] = True
        if keys[pygame.K_e]: well_states[2] = False

        # Convert desired state to a single action for the step function
        # This is a simplified mapping for human playability
        current_well_states = [w['active'] for w in env.wells]
        if well_states[0] and not current_well_states[0]: action[0] = 1 # Activate Red
        elif not well_states[0] and current_well_states[0]: action[0] = 4 # Deactivate Red
        
        if well_states[1] and not current_well_states[1]: action[0] = 2 # Activate Green
        elif not well_states[1] and current_well_states[1]: action[1] = 1 # Deactivate Green
        
        if well_states[2] and not current_well_states[2]: action[0] = 3 # Activate Blue
        elif not well_states[2] and current_well_states[2]: action[2] = 1 # Deactivate Blue
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                terminated = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)

    env.close()