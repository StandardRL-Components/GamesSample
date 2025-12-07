
# Generated: 2025-08-27T12:30:32.794567
# Source Brief: brief_00068.md
# Brief Index: 68

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to select a plot. Hold Space to plant a seed, or Shift to harvest a mature crop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage your isometric farm. Plant seeds, wait for them to grow, and harvest for coins. Reach 1000 coins before the 5-minute timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 6, 6
    WIN_COIN_TARGET = 1000
    MAX_TIME_SECONDS = 300  # 5 minutes
    FPS = 30

    # Colors
    COLOR_BG = (58, 79, 65)
    COLOR_SOIL = (110, 65, 43)
    COLOR_SOIL_DARK = (87, 51, 34)
    COLOR_PLANTED = (124, 252, 0)
    COLOR_GROWN = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (255, 248, 220)
    COLOR_UI_SHADOW = (40, 40, 40)

    # Plot States
    STATE_EMPTY = 0
    STATE_PLANTED = 1
    STATE_GROWN = 2

    # Timings
    GROWTH_TIME_SECONDS = 5
    ACTION_COOLDOWN_FRAMES = 5  # Cooldown between plant/harvest actions
    MOVE_COOLDOWN_FRAMES = 4    # Cooldown for cursor movement

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Isometric projection constants
        self.tile_width_half = 48
        self.tile_height_half = 24
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 120

        # Pre-calculate tile points for performance
        self.tile_points = [
            (0, -self.tile_height_half),
            (self.tile_width_half, 0),
            (0, self.tile_height_half),
            (-self.tile_width_half, 0),
        ]

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.time_left = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.action_cooldown = 0
        self.move_cooldown = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.grid = [
            [{'state': self.STATE_EMPTY, 'growth': 0} for _ in range(self.GRID_COLS)]
            for _ in range(self.GRID_ROWS)
        ]
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.score = 0
        self.time_left = self.MAX_TIME_SECONDS
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.action_cooldown = 0
        self.move_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        self._update_timers()
        self._handle_movement(movement)
        
        if self.action_cooldown == 0:
            if space_held:
                reward += self._action_plant()
            elif shift_held:
                reward += self._action_harvest()

        self._update_crops()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.score >= self.WIN_COIN_TARGET:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_timers(self):
        self.time_left = max(0, self.time_left - 1 / self.FPS)
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

    def _handle_movement(self, movement):
        if self.move_cooldown > 0 or movement == 0:
            return

        r, c = self.cursor_pos
        if movement == 1: r -= 1  # Up
        elif movement == 2: r += 1  # Down
        elif movement == 3: c -= 1  # Left
        elif movement == 4: c += 1  # Right
        
        if 0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS:
            self.cursor_pos = [r, c]
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

    def _action_plant(self):
        r, c = self.cursor_pos
        plot = self.grid[r][c]
        if plot['state'] == self.STATE_EMPTY:
            plot['state'] = self.STATE_PLANTED
            plot['growth'] = 0
            self.action_cooldown = self.ACTION_COOLDOWN_FRAMES
            # SFX: plant_seed.wav
            self._create_particles(r, c, self.COLOR_PLANTED, 10, 2)
            return 0.1
        return 0

    def _action_harvest(self):
        r, c = self.cursor_pos
        plot = self.grid[r][c]
        if plot['state'] == self.STATE_GROWN:
            crop_value = self.np_random.integers(15, 26)
            self.score += crop_value
            plot['state'] = self.STATE_EMPTY
            plot['growth'] = 0
            self.action_cooldown = self.ACTION_COOLDOWN_FRAMES
            # SFX: cash_register.wav
            self._create_particles(r, c, self.COLOR_GROWN, 20, 4, fly_to_ui=True)
            return 10.0
        return 0

    def _update_crops(self):
        growth_per_frame = 1 / (self.GROWTH_TIME_SECONDS * self.FPS)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot = self.grid[r][c]
                if plot['state'] == self.STATE_PLANTED:
                    plot['growth'] += growth_per_frame
                    if plot['growth'] >= 1.0:
                        plot['state'] = self.STATE_GROWN
                        plot['growth'] = 1.0
                        # SFX: crop_ready.wav
                        self._create_particles(r, c, self.COLOR_GROWN, 5, 1, is_sparkle=True)

    def _check_termination(self):
        if self.score >= self.WIN_COIN_TARGET or self.time_left <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left}
    
    def _grid_to_screen(self, r, c):
        x = self.origin_x + (c - r) * self.tile_width_half
        y = self.origin_y + (c + r) * self.tile_height_half
        return int(x), int(y)

    def _render_game(self):
        # Draw plots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                screen_x, screen_y = self._grid_to_screen(r, c)
                points = [(screen_x + px, screen_y + py) for px, py in self.tile_points]
                
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SOIL)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SOIL_DARK)

                plot = self.grid[r][c]
                if plot['state'] == self.STATE_PLANTED:
                    self._draw_crop(screen_x, screen_y, plot['growth'], self.COLOR_PLANTED)
                elif plot['state'] == self.STATE_GROWN:
                    self._draw_crop(screen_x, screen_y, 1.0, self.COLOR_GROWN)
        
        # Draw particles
        self._draw_particles()

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        screen_x, screen_y = self._grid_to_screen(cursor_r, cursor_c)
        points = [(screen_x + px, screen_y + py) for px, py in self.tile_points]
        
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        cursor_color = (
            int(128 + 127 * pulse),
            int(128 + 127 * pulse),
            int(128 + 127 * pulse)
        )
        pygame.draw.aalines(self.screen, cursor_color, True, points, 2)


    def _draw_crop(self, x, y, growth_progress, color):
        max_height = 30
        max_width = 15
        
        height = int(max_height * growth_progress)
        width = int(max_width * growth_progress)
        
        if height <= 0 or width <= 0:
            return

        # Simple diamond shape for crop
        points = [
            (x, y - height),
            (x + width, y),
            (x, y + height / 3), # Give it some base
            (x - width, y),
        ]
        
        darker_color = tuple(max(0, val - 40) for val in color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, darker_color)


    def _render_ui(self):
        # --- Score Display ---
        score_text = f"{self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_UI_TEXT)
        score_shadow = self.font_large.render(score_text, True, self.COLOR_UI_SHADOW)
        score_pos = (self.SCREEN_WIDTH - score_surf.get_width() - 20, 18)
        self.screen.blit(score_shadow, (score_pos[0] + 2, score_pos[1] + 2))
        self.screen.blit(score_surf, score_pos)
        # Coin Icon
        pygame.gfxdraw.filled_circle(self.screen, score_pos[0] - 20, score_pos[1] + 20, 10, self.COLOR_GROWN)
        pygame.gfxdraw.aacircle(self.screen, score_pos[0] - 20, score_pos[1] + 20, 10, self.COLOR_UI_SHADOW)

        # --- Timer Display ---
        minutes = int(self.time_left // 60)
        seconds = int(self.time_left % 60)
        time_text = f"{minutes:02}:{seconds:02}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        time_shadow = self.font_large.render(time_text, True, self.COLOR_UI_SHADOW)
        time_pos = (20, 18)
        self.screen.blit(time_shadow, (time_pos[0] + 2, time_pos[1] + 2))
        self.screen.blit(time_surf, time_pos)

        # --- Game Over/Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_COIN_TARGET:
                msg = "YOU WIN!"
            else:
                msg = "TIME'S UP!"
            
            msg_surf = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            msg_shadow = self.font_large.render(msg, True, self.COLOR_UI_SHADOW)
            msg_pos = (
                self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2,
                self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2
            )
            self.screen.blit(msg_shadow, (msg_pos[0] + 2, msg_pos[1] + 2))
            self.screen.blit(msg_surf, msg_pos)
    
    def _create_particles(self, r, c, color, count, speed, fly_to_ui=False, is_sparkle=False):
        x, y = self._grid_to_screen(r, c)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_x = math.cos(angle) * self.np_random.uniform(0.5, 1.5) * speed
            vel_y = math.sin(angle) * self.np_random.uniform(0.5, 1.5) * speed
            
            if is_sparkle:
                vel_y = -abs(vel_y) # Sparkles go up

            particle = {
                'pos': [x, y],
                'vel': [vel_x, vel_y],
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.integers(2, 5),
                'fly_to_ui': fly_to_ui
            }
            self.particles.append(particle)

    def _update_particles(self):
        ui_target = (self.SCREEN_WIDTH - 40, 40)
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)
                continue
            
            if p['fly_to_ui']:
                # Move towards the UI score element
                target_vec = (ui_target[0] - p['pos'][0], ui_target[1] - p['pos'][1])
                dist = math.hypot(*target_vec)
                if dist < 10:
                    self.particles.pop(i)
                    continue
                p['vel'][0] = p['vel'][0] * 0.85 + target_vec[0] * 0.1
                p['vel'][1] = p['vel'][1] * 0.85 + target_vec[1] * 0.1
            else:
                 p['vel'][1] += 0.1 # Gravity
            
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] = max(0, p['radius'] * 0.95)

    def _draw_particles(self):
        for p in self.particles:
            if p['radius'] >= 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color']
                )

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to interact with the game
    render_mode = "human_playable" # "rgb_array" or "human_playable"

    if render_mode == "human_playable":
        # Modify the class to add human rendering
        class HumanPlayableGameEnv(GameEnv):
            def __init__(self, render_mode="human"):
                super().__init__(render_mode)
                self.metadata["render_modes"].append("human")
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Farm Manager")

            def _get_observation(self):
                # In human mode, we render to screen then copy to array
                super()._get_observation() # Renders to self.screen surface
                if self.render_mode == "human":
                    pygame.display.flip()
                arr = pygame.surfarray.array3d(self.screen)
                return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
        env = HumanPlayableGameEnv(render_mode="human")
    else:
        env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Human Interaction Loop ---
    if render_mode == "human_playable":
        print(env.game_description)
        print(env.user_guide)
        
        action = env.action_space.sample()
        action.fill(0) # Start with no-op

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            
            # Reset action
            action.fill(0)

            # Movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            else: action[0] = 0

            # Buttons
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                done = True

            env.clock.tick(env.FPS)
            pygame.display.set_caption(f"Farm Manager | Score: {info['score']} | Time: {info['time_left']:.0f}")

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    # --- Standard Gym Loop (for testing) ---
    else:
        for _ in range(5000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                obs, info = env.reset()
                total_reward = 0

    env.close()