import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:28:38.519250
# Source Brief: brief_01089.md
# Brief Index: 1089
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent controls three gravity wells to capture asteroids.
    
    The agent's goal is to accumulate 1000 units of mass by capturing asteroids before
    the episode ends. The primary challenge is managing the strength of the three wells,
    which constantly decay over time.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement/Well Control
        - 0 (none): No-op.
        - 1 (up): Increase strength of the selected well.
        - 2 (down): Decrease strength of the selected well.
        - 3 (left): Select the previous well.
        - 4 (right): Select the next well.
    - actions[1]: Space button (unused).
    - actions[2]: Shift button (unused).

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.

    Reward Structure:
    - + (mass * 1.0) for each captured asteroid.
    - +100 for winning the game (reaching 1000 mass).
    - Small positive reward for pulling asteroids closer to wells.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three gravity wells to capture asteroids and collect their mass. "
        "Adjust the strength of each well to create a stable orbit and prevent asteroids from escaping."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a gravity well. Use ↑↓ arrow keys to "
        "increase or decrease the selected well's strength."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 1000
    MAX_STEPS = 2000
    
    # --- Colors ---
    COLOR_BG = (10, 10, 26)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WELL = (0, 170, 255)
    COLOR_WELL_SELECTED = (255, 255, 255)
    COLOR_ASTEROID_CAPTURED = (0, 255, 136)
    
    # --- Game Physics & Parameters ---
    GRAVITATIONAL_CONSTANT = 700.0
    WELL_DECAY_PER_STEP = 10.0 / FPS  # 10 strength per second
    WELL_ADJUST_AMOUNT = 10.0
    CAPTURE_VELOCITY_THRESHOLD = 1.5
    MIN_ASTEROIDS = 15
    MAX_ASTEROIDS = 25
    EDGE_REPULSION_FORCE = 2000.0
    EDGE_REPULSION_MARGIN = 30
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.wells = []
        self.asteroids = []
        self.particles = []
        self.selected_well_index = 0
        self.last_action_feedback = {}

        # self.reset() is called by the wrapper, no need to call it here
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_well_index = 0
        self.last_action_feedback.clear()

        # --- Initialize Wells ---
        self.wells = [
            {'pos': pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.5), 'strength': 50.0},
            {'pos': pygame.Vector2(self.WIDTH * 0.50, self.HEIGHT * 0.25), 'strength': 50.0},
            {'pos': pygame.Vector2(self.WIDTH * 0.75, self.HEIGHT * 0.75), 'strength': 50.0},
        ]
        
        # --- Initialize Asteroids ---
        self.asteroids = []
        num_asteroids = self.np_random.integers(self.MIN_ASTEROIDS, self.MAX_ASTEROIDS + 1)
        for _ in range(num_asteroids):
            self._spawn_asteroid()

        # --- Initialize Particles ---
        self.particles = []
        # Create a persistent starfield
        for _ in range(100):
            self.particles.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'vel': pygame.Vector2(0, 0),
                'lifespan': float('inf'), # a star
                'color': (self.np_random.integers(50, 100),) * 3,
                'radius': self.np_random.uniform(0.5, 1.5)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Store previous state for reward calculation ---
        prev_asteroid_dists = self._get_asteroid_distances()

        # --- Handle Player Action ---
        self._handle_action(action)

        # --- Update Game Logic ---
        self._update_wells()
        self._update_asteroids()
        self._update_particles()
        
        # --- Calculate Reward & Check for Termination ---
        capture_reward = self._check_captures()
        reward += capture_reward
        
        # Continuous reward for pulling asteroids closer
        current_asteroid_dists = self._get_asteroid_distances()
        for i, asteroid in enumerate(self.asteroids):
            asteroid_id = asteroid.get('id', i)
            if asteroid_id in prev_asteroid_dists:
                prev_dists = prev_asteroid_dists[asteroid_id]
                current_dists = current_asteroid_dists.get(asteroid_id)
                if current_dists:
                    for j in range(len(self.wells)):
                        # Reward is positive if distance decreased
                        dist_delta = prev_dists[j] - current_dists[j]
                        if dist_delta > 0:
                            # Scale by mass and a small constant
                            reward += 0.001 * asteroid['mass'] * dist_delta
        
        terminated = self._check_termination()
        
        if terminated and self.score >= self.WIN_SCORE:
            reward += 100.0 # Goal-oriented reward

        # Truncation condition
        truncated = self.steps >= self.MAX_STEPS

        # Clamp reward
        reward = np.clip(reward, -10.0, 100.0)

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, _, _ = action # space_held, shift_held are unused

        if movement == 1: # Up: Increase strength
            well = self.wells[self.selected_well_index]
            well['strength'] = min(100.0, well['strength'] + self.WELL_ADJUST_AMOUNT)
            self._create_action_feedback(well['pos'], self.COLOR_WELL_SELECTED, '+')
        elif movement == 2: # Down: Decrease strength
            well = self.wells[self.selected_well_index]
            well['strength'] = max(0.0, well['strength'] - self.WELL_ADJUST_AMOUNT)
            self._create_action_feedback(well['pos'], self.COLOR_WELL_SELECTED, '-')
        elif movement == 3: # Left: Previous well
            self.selected_well_index = (self.selected_well_index - 1) % len(self.wells)
        elif movement == 4: # Right: Next well
            self.selected_well_index = (self.selected_well_index + 1) % len(self.wells)

    def _update_wells(self):
        for well in self.wells:
            if well['strength'] > 0:
                well['strength'] = max(0.0, well['strength'] - self.WELL_DECAY_PER_STEP)

    def _update_asteroids(self):
        for i, asteroid in enumerate(self.asteroids):
            net_force = pygame.Vector2(0, 0)
            
            # Gravity from wells
            for well in self.wells:
                vec_to_well = well['pos'] - asteroid['pos']
                dist_sq = vec_to_well.length_squared()
                if dist_sq > 1: # Avoid division by zero
                    force_magnitude = (self.GRAVITATIONAL_CONSTANT * well['strength'] * asteroid['mass']) / dist_sq
                    net_force += vec_to_well.normalize() * force_magnitude
            
            # Repulsion from edges
            if asteroid['pos'].x < self.EDGE_REPULSION_MARGIN:
                net_force.x += self.EDGE_REPULSION_FORCE / max(1, asteroid['pos'].x)
            if asteroid['pos'].x > self.WIDTH - self.EDGE_REPULSION_MARGIN:
                net_force.x -= self.EDGE_REPULSION_FORCE / max(1, self.WIDTH - asteroid['pos'].x)
            if asteroid['pos'].y < self.EDGE_REPULSION_MARGIN:
                net_force.y += self.EDGE_REPULSION_FORCE / max(1, asteroid['pos'].y)
            if asteroid['pos'].y > self.HEIGHT - self.EDGE_REPULSION_MARGIN:
                net_force.y -= self.EDGE_REPULSION_FORCE / max(1, self.HEIGHT - asteroid['pos'].y)

            # Update physics (F=ma -> a = F/m)
            acceleration = net_force / asteroid['mass']
            asteroid['vel'] += acceleration * (1/self.FPS)
            
            # Apply drag
            asteroid['vel'] *= 0.99
            
            asteroid['pos'] += asteroid['vel']
            asteroid['id'] = i # Ensure ID is up-to-date for reward calculation

    def _check_captures(self):
        captured_reward = 0
        asteroids_to_remove = []
        for i, asteroid in reversed(list(enumerate(self.asteroids))):
            for well in self.wells:
                dist_to_well = asteroid['pos'].distance_to(well['pos'])
                well_radius = well['strength'] * 0.5 # Effective radius
                
                if dist_to_well < well_radius and asteroid['vel'].length() < self.CAPTURE_VELOCITY_THRESHOLD:
                    # Capture!
                    self.score += asteroid['mass']
                    captured_reward += asteroid['mass'] * 1.0 # Event-based reward
                    self._create_capture_particles(asteroid['pos'], asteroid['mass'])
                    asteroids_to_remove.append(i)
                    # Sfx: Play capture sound
                    break # Asteroid can only be captured once
        
        if asteroids_to_remove:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]
            # Spawn new asteroids to replace captured ones
            for _ in range(len(asteroids_to_remove)):
                self._spawn_asteroid()

        return captured_reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        # Terminate if all asteroids are gone (unlikely with continuous spawning, but a safeguard)
        if not self.asteroids and self.steps > 1:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles (stars and effects)
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color']
            )

        # Render wells
        for i, well in enumerate(self.wells):
            radius = int(well['strength'] * 0.5)
            if radius > 1:
                # Pulsating glow effect
                glow_alpha = 30 + 30 * math.sin(self.steps * 0.1 + i)
                glow_color = (*self.COLOR_WELL[:3], glow_alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(well['pos'].x), int(well['pos'].y), int(radius * 1.5), glow_color)
                
                # Main well circle
                pygame.gfxdraw.aacircle(self.screen, int(well['pos'].x), int(well['pos'].y), radius, self.COLOR_WELL)
                pygame.gfxdraw.filled_circle(self.screen, int(well['pos'].x), int(well['pos'].y), radius, (*self.COLOR_WELL[:3], 50))
        
        # Render asteroids
        for asteroid in self.asteroids:
            mass_color_val = int(100 + (asteroid['mass'] - 1) * (155 / 9))
            color = (mass_color_val, mass_color_val, mass_color_val)
            radius = int(2 + asteroid['mass'] * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(asteroid['pos'].x), int(asteroid['pos'].y), radius, (255,255,255,50))

    def _render_ui(self):
        # Render score
        score_text = self.font_main.render(f"MASS: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Render step counter
        step_text = self.font_small.render(f"STEP: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (10, 40))

        # Render selected well indicator
        if self.wells:
            selected_well = self.wells[self.selected_well_index]
            indicator_pos = selected_well['pos'] + pygame.Vector2(0, -selected_well['strength']*0.5 - 15)
            points = [
                (indicator_pos.x, indicator_pos.y),
                (indicator_pos.x - 5, indicator_pos.y + 10),
                (indicator_pos.x + 5, indicator_pos.y + 10)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_WELL_SELECTED)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_WELL_SELECTED)

        # Render action feedback
        for key, fb in list(self.last_action_feedback.items()):
            fb['lifespan'] -= 1
            if fb['lifespan'] <= 0:
                del self.last_action_feedback[key]
            else:
                alpha = int(255 * (fb['lifespan'] / fb['max_lifespan']))
                try:
                    color = (*fb['color'][:3], alpha)
                    feedback_text = self.font_main.render(fb['text'], True, color)
                    text_rect = feedback_text.get_rect(center=fb['pos'])
                    self.screen.blit(feedback_text, text_rect)
                except (ValueError, TypeError):
                     # Handle potential color format issues gracefully
                     pass


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wells_strength": [w['strength'] for w in self.wells],
            "num_asteroids": len(self.asteroids),
        }

    def close(self):
        pygame.quit()

    # --- Helper Methods ---
    def _spawn_asteroid(self):
        # Spawn at edges
        side = self.np_random.integers(0, 4)
        if side == 0: # top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -10)
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(0.5, 2))
        elif side == 1: # bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-2, -0.5))
        elif side == 2: # left
            pos = pygame.Vector2(-10, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.Vector2(self.np_random.uniform(0.5, 2), self.np_random.uniform(-1, 1))
        else: # right
            pos = pygame.Vector2(self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.Vector2(self.np_random.uniform(-2, -0.5), self.np_random.uniform(-1, 1))
            
        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'mass': self.np_random.integers(1, 11),
            'id': len(self.asteroids)
        })

    def _create_capture_particles(self, pos, mass):
        num_particles = int(mass * 3)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'lifespan': self.np_random.integers(15, 30),
                'color': self.COLOR_ASTEROID_CAPTURED,
                'radius': self.np_random.uniform(1, 3)
            })

    def _create_action_feedback(self, pos, color, text):
        key = (pos.x, pos.y)
        self.last_action_feedback[key] = {
            'pos': pos + pygame.Vector2(0, -40),
            'text': text,
            'color': color,
            'lifespan': 15, # frames
            'max_lifespan': 15
        }

    def _update_particles(self):
        # Exclude stars from updates
        active_particles = [p for p in self.particles if p['lifespan'] != float('inf')]
        
        for p in active_particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95 # shrink
            p['vel'] *= 0.98 # slow down

        # Remove dead particles (but keep stars)
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_asteroid_distances(self):
        """Returns a dict {asteroid_id: [dist_to_well_0, dist_to_well_1, ...]}"""
        dists = {}
        for i, asteroid in enumerate(self.asteroids):
            asteroid_dists = [asteroid['pos'].distance_to(w['pos']) for w in self.wells]
            dists[asteroid.get('id', i)] = asteroid_dists
        return dists

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to set SDL_VIDEODRIVER to a real driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gravity Well")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit | R: Reset")
    
    # Use a dictionary to track key presses for single action events
    key_action_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    last_key_action = 0

    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r: # Reset env
                    obs, info = env.reset()
                    total_reward = 0
                if event.key in key_action_map:
                    last_key_action = key_action_map[event.key]

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Apply single-press actions
        if last_key_action != 0:
            action[0] = last_key_action
            last_key_action = 0 # Reset after applying
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.FPS)
        
    env.close()