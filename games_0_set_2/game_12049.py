import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:00:28.034134
# Source Brief: brief_02049.md
# Brief Index: 2049
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls four oscillating magnets
    to attract and collect swirling metal flakes. The goal is to collect
    70% of the flakes within a 30-second time limit.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement):
        - 0: No-op
        - 1: Move Red Magnet UP
        - 2: Move Green Magnet DOWN
        - 3: Move Blue Magnet LEFT
        - 4: Move Yellow Magnet RIGHT
    - action[1] (Space): 0=released, 1=held (Activates magnets)
    - action[2] (Shift): 0=released, 1=held (Reverses magnet polarity)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 for each flake collected.
    - -0.01 per step with no collection.
    - +100 for winning (collecting 70% of flakes).
    - -10 for losing (time runs out before reaching 70%).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control four oscillating magnets to attract and collect swirling metal flakes. "
        "Collect 70% of the flakes before time runs out to win."
    )
    user_guide = (
        "Use directional keys to move the corresponding magnets (e.g., ↑ for Red, ↓ for Green). "
        "Hold space to activate magnets and shift to reverse their polarity."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    TOTAL_FLAKES = 200
    WIN_PERCENTAGE = 0.7
    WIN_CONDITION_COUNT = int(TOTAL_FLAKES * WIN_PERCENTAGE)

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_FLAKE = (255, 255, 255)
    COLOR_FLAKE_COLLECTED = (80, 80, 90)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PROGRESS_BAR_BG = (40, 50, 70)
    COLOR_PROGRESS_BAR_FG = (100, 140, 255)
    COLOR_WIN_TEXT = (100, 255, 100)
    COLOR_LOSS_TEXT = (255, 100, 100)

    MAGNET_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80)   # Yellow
    ]

    # Gameplay
    MAGNET_SPEED = 5
    MAGNET_BASE_STRENGTH = 25000
    MAGNET_RADIUS = 15
    FLAKE_FRICTION = 0.98
    COLLECTION_RADIUS = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 16)
        self.font_end = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables are initialized in reset()
        self.magnets = []
        self.flakes = []
        self.steps = 0
        self.score = 0
        self.collected_flakes_count = 0
        self.game_over = False
        self.win_status = False
        self.are_magnets_active = False
        self.is_polarity_reversed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.collected_flakes_count = 0
        self.game_over = False
        self.win_status = False
        self.are_magnets_active = False
        self.is_polarity_reversed = False

        # Initialize Magnets
        self.magnets = [
            {'pos': np.array([100.0, 100.0]), 'color': self.MAGNET_COLORS[0]}, # Red
            {'pos': np.array([self.WIDTH - 100.0, self.HEIGHT - 100.0]), 'color': self.MAGNET_COLORS[1]}, # Green
            {'pos': np.array([100.0, self.HEIGHT - 100.0]), 'color': self.MAGNET_COLORS[2]}, # Blue
            {'pos': np.array([self.WIDTH - 100.0, 100.0]), 'color': self.MAGNET_COLORS[3]}  # Yellow
        ]

        # Initialize Flakes
        self.flakes = []
        for _ in range(self.TOTAL_FLAKES):
            self.flakes.append({
                'pos': np.array([
                    self.np_random.uniform(20, self.WIDTH - 20),
                    self.np_random.uniform(20, self.HEIGHT - 20)
                ]),
                'vel': np.array([0.0, 0.0]),
                'collected': False,
                'size': self.np_random.uniform(1.0, 2.5)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        self.are_magnets_active = space_held == 1
        self.is_polarity_reversed = shift_held == 1
        
        self._handle_input(movement)
        
        flakes_collected_this_step = self._update_physics_and_collection()
        
        self.steps += 1
        
        reward = self._calculate_reward(flakes_collected_this_step)
        self.score += reward

        terminated = self._check_termination()
        
        if terminated:
            self.game_over = True
            # Apply terminal reward
            if self.win_status:
                self.score += 100
                reward += 100
            else: # Time ran out
                self.score -= 10
                reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # 1=Up (Red), 2=Down (Green), 3=Left (Blue), 4=Right (Yellow)
        if movement == 1: self.magnets[0]['pos'][1] -= self.MAGNET_SPEED
        elif movement == 2: self.magnets[1]['pos'][1] += self.MAGNET_SPEED
        elif movement == 3: self.magnets[2]['pos'][0] -= self.MAGNET_SPEED
        elif movement == 4: self.magnets[3]['pos'][0] += self.MAGNET_SPEED
        
        # Clamp magnet positions to screen bounds
        for magnet in self.magnets:
            magnet['pos'][0] = np.clip(magnet['pos'][0], self.MAGNET_RADIUS, self.WIDTH - self.MAGNET_RADIUS)
            magnet['pos'][1] = np.clip(magnet['pos'][1], self.MAGNET_RADIUS, self.HEIGHT - self.MAGNET_RADIUS)

    def _update_physics_and_collection(self):
        flakes_collected_this_step = 0
        
        # Get current magnet strength from oscillation
        # Sinusoidal wave with a 10-second period
        oscillation_phase = (self.steps / self.FPS) * (2 * math.pi / 10)
        oscillation_multiplier = 1.0 + 0.5 * math.sin(oscillation_phase)
        
        current_strength = self.MAGNET_BASE_STRENGTH * oscillation_multiplier
        if self.is_polarity_reversed:
            current_strength *= -1 # Repel when attracting, attract when repelling

        for flake in self.flakes:
            if flake['collected']:
                continue

            # --- Physics Update ---
            if self.are_magnets_active:
                total_force = np.array([0.0, 0.0])
                for magnet in self.magnets:
                    vec_to_magnet = magnet['pos'] - flake['pos']
                    dist_sq = np.sum(vec_to_magnet**2)
                    if dist_sq < 1: dist_sq = 1 # Avoid division by zero
                    
                    force_magnitude = current_strength / dist_sq
                    force_vector = (vec_to_magnet / np.sqrt(dist_sq)) * force_magnitude
                    total_force += force_vector
                
                flake['vel'] += total_force / (flake['size']**2) # Heavier flakes are less affected

            flake['vel'] *= self.FLAKE_FRICTION
            flake['pos'] += flake['vel']

            # Clamp flake positions
            flake['pos'][0] = np.clip(flake['pos'][0], 0, self.WIDTH)
            flake['pos'][1] = np.clip(flake['pos'][1], 0, self.HEIGHT)

            # --- Collection Check ---
            # Flakes can only be collected when magnets are active and attracting
            if self.are_magnets_active and current_strength > 0:
                for magnet in self.magnets:
                    if np.linalg.norm(magnet['pos'] - flake['pos']) < self.COLLECTION_RADIUS:
                        flake['collected'] = True
                        flake['vel'] = np.array([0.0, 0.0]) # Stop moving
                        self.collected_flakes_count += 1
                        flakes_collected_this_step += 1
                        # SFX: collection_chime.wav
                        break # Flake collected, move to next flake
        
        return flakes_collected_this_step

    def _calculate_reward(self, flakes_collected_this_step):
        if flakes_collected_this_step > 0:
            return 0.1 * flakes_collected_this_step
        else:
            return -0.01

    def _check_termination(self):
        if self.collected_flakes_count >= self.WIN_CONDITION_COUNT:
            self.win_status = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.win_status = False
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_flakes": self.collected_flakes_count,
            "win": self.win_status
        }

    def _render_game(self):
        # Render Flakes
        for flake in self.flakes:
            color = self.COLOR_FLAKE_COLLECTED if flake['collected'] else self.COLOR_FLAKE
            pos = (int(flake['pos'][0]), int(flake['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(flake['size']), color)

        # Render Magnets
        oscillation_phase = (self.steps / self.FPS) * (2 * math.pi / 10)
        oscillation_multiplier = 1.0 + 0.5 * math.sin(oscillation_phase)
        
        for magnet in self.magnets:
            pos = (int(magnet['pos'][0]), int(magnet['pos'][1]))
            
            # Draw glow effect if active
            if self.are_magnets_active:
                # SFX: magnet_hum_loop.wav
                glow_radius = int(self.MAGNET_RADIUS * 1.8 * (0.5 + oscillation_multiplier / 2))
                
                # Polarity indication in glow
                glow_color_base = self.COLOR_FLAKE if self.is_polarity_reversed else magnet['color']
                
                # Fade glow from center
                for i in range(glow_radius, 0, -2):
                    alpha = 60 * (1 - i / glow_radius)
                    glow_color = (*glow_color_base, alpha)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, glow_color)
            
            # Draw main magnet circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.MAGNET_RADIUS, magnet['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.MAGNET_RADIUS, magnet['color'])

    def _render_ui(self):
        # Progress Bar
        progress = self.collected_flakes_count / self.WIN_CONDITION_COUNT
        progress = min(progress, 1.0)
        bar_width = self.WIDTH - 40
        bar_height = 20
        
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (20, 15, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_FG, (20, 15, int(bar_width * progress), bar_height), border_radius=5)
        
        # Percentage Text
        percent_text = f"COLLECTED: {self.collected_flakes_count}/{self.TOTAL_FLAKES} ({int(self.collected_flakes_count/self.TOTAL_FLAKES*100)}%)"
        text_surface = self.font_main.render(percent_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (25, 16))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        timer_text = f"TIME: {time_left:.1f}s"
        timer_surface = self.font_timer.render(timer_text, True, self.COLOR_TEXT)
        timer_rect = timer_surface.get_rect(topright=(self.WIDTH - 20, 18))
        self.screen.blit(timer_surface, timer_rect)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.win_status:
            text = "VICTORY"
            color = self.COLOR_WIN_TEXT
        else:
            text = "TIME UP"
            color = self.COLOR_LOSS_TEXT
            
        text_surface = self.font_end.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        overlay.blit(text_surface, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert self.collected_flakes_count == 0
        assert len(self.flakes) == self.TOTAL_FLAKES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == '__main__':
    # This block is for human play and will not run in the headless environment
    # It requires a display to be available.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Magnetic Swirl")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        # The user_guide suggests arrow keys, but the example code uses WASD.
        # This mapping corresponds to the action space description.
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1 # Red Up
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2 # Green Down
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3 # Blue Left
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4 # Yellow Right
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Final score: {info['score']:.2f}, Win: {info['win']}")
            # Wait for 'R' to reset
            
        clock.tick(GameEnv.FPS)
        
    env.close()