
# Generated: 2025-08-28T01:39:58.207696
# Source Brief: brief_04186.md
# Brief Index: 4186

        
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


# Helper class for particle effects
class Particle:
    def __init__(self, x, y, color, lifetime, vx_range, vy_range, size_range, gravity=0.1):
        self.x = x
        self.y = y
        self.vx = random.uniform(*vx_range)
        self.vy = random.uniform(*vy_range)
        self.gravity = gravity
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.size = random.uniform(*size_range)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
            surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a plot. Press Space to plant a seed or harvest a ready crop. Press Shift to sell all harvested crops."
    )

    game_description = (
        "A cheerful, fast-paced farming simulation. Plant, grow, and harvest crops to earn 1000 coins before the sun sets!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 6
        self.PLOT_SIZE = 40
        self.GRID_MARGIN_X = (self.WIDTH - self.GRID_COLS * self.PLOT_SIZE) // 2
        self.GRID_MARGIN_Y = (self.HEIGHT - self.GRID_ROWS * self.PLOT_SIZE) // 2 + 20

        # Game constants
        self.MAX_TIME = 1800  # 60 seconds at 30fps
        self.MAX_STEPS = 2000 # Safety limit
        self.WIN_SCORE = 1000
        self.CROP_TYPES = {
            'carrot': {'grow_time': 240, 'price': 15, 'color': (255, 140, 0), 'seed_color': (100, 50, 0)},
            'lettuce': {'grow_time': 150, 'price': 8, 'color': (144, 238, 144), 'seed_color': (50, 100, 50)},
        }

        # Colors
        self.COLOR_BG = (87, 138, 52)
        self.COLOR_SOIL = (94, 65, 41)
        self.COLOR_SOIL_WET = (74, 45, 21)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 220)
        self.COLOR_GOLD = (255, 215, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_large = pygame.font.SysFont(None, 52)

        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.win_condition_met = False

        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.plots = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self.plots.append({
                    'state': 'empty',  # 'empty', 'growing', 'ready'
                    'growth': 0.0,
                    'type': None,
                })

        self.harvested_crops = {name: 0 for name in self.CROP_TYPES}
        self.particles = []

        self.last_space_held = False
        self.last_shift_held = False
        self.last_reward_score_milestone = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.time_remaining -= 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        # --- Update Game Logic ---
        reward += self._handle_input(movement, space_press, shift_press)
        self._update_world()
        self._update_particles()
        
        # Calculate score-based reward
        new_milestone = self.score // 10
        if new_milestone > self.last_reward_score_milestone:
            reward += (new_milestone - self.last_reward_score_milestone)
            self.last_reward_score_milestone = new_milestone

        # Check termination conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            if not self.win_condition_met: # Grant reward only once
                reward += 100
                self.win_condition_met = True
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            if not self.win_condition_met:
                reward -= 100
            terminated = True
            self.game_over = True

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        reward = 0
        # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.selector_pos[1] -= 1
        elif movement == 2: self.selector_pos[1] += 1
        elif movement == 3: self.selector_pos[0] -= 1
        elif movement == 4: self.selector_pos[0] += 1
        
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_COLS - 1)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_ROWS - 1)

        # Action: Plant/Harvest
        if space_press:
            idx = self.selector_pos[1] * self.GRID_COLS + self.selector_pos[0]
            plot = self.plots[idx]
            if plot['state'] == 'empty':
                # Plant a random crop
                plot['state'] = 'growing'
                plot['type'] = self.np_random.choice(list(self.CROP_TYPES.keys()))
                plot['growth'] = 0.0
                reward += 0.1
                # # Sound: plant_seed.wav
                self._create_particles(self.selector_pos, 10, self.COLOR_SOIL, (1, 3))
            elif plot['state'] == 'ready':
                # Harvest
                self.harvested_crops[plot['type']] += 1
                plot['state'] = 'empty'
                plot['type'] = None
                reward += 0.2
                # # Sound: harvest.wav
                self._create_particles(self.selector_pos, 15, self.COLOR_GOLD, (2, 4))
        
        # Action: Sell
        if shift_press:
            money_earned = 0
            for crop_type, count in self.harvested_crops.items():
                if count > 0:
                    money_earned += count * self.CROP_TYPES[crop_type]['price']
            
            if money_earned > 0:
                self.score += money_earned
                self.harvested_crops = {name: 0 for name in self.CROP_TYPES}
                # # Sound: cash_register.wav
                self._create_coin_particles(20)

        return reward

    def _update_world(self):
        for plot in self.plots:
            if plot['state'] == 'growing':
                grow_time = self.CROP_TYPES[plot['type']]['grow_time']
                plot['growth'] += 1.0 / max(1, grow_time)
                if plot['growth'] >= 1.0:
                    plot['growth'] = 1.0
                    plot['state'] = 'ready'
                    # # Sound: crop_ready.wav

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw plots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                idx = r * self.GRID_COLS + c
                plot = self.plots[idx]
                rect = pygame.Rect(
                    self.GRID_MARGIN_X + c * self.PLOT_SIZE,
                    self.GRID_MARGIN_Y + r * self.PLOT_SIZE,
                    self.PLOT_SIZE, self.PLOT_SIZE
                )
                
                # Draw soil
                soil_color = self.COLOR_SOIL_WET if plot['state'] != 'empty' else self.COLOR_SOIL
                pygame.draw.rect(self.screen, soil_color, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

                # Draw crop
                if plot['state'] == 'growing':
                    crop_info = self.CROP_TYPES[plot['type']]
                    lerp_color = tuple(int(sc + (fc - sc) * plot['growth']) for sc, fc in zip(crop_info['seed_color'], crop_info['color']))
                    radius = int((self.PLOT_SIZE * 0.4) * plot['growth'])
                    pos = (rect.centerx, rect.centery)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], max(1, radius), lerp_color)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(1, radius), lerp_color)
                elif plot['state'] == 'ready':
                    crop_info = self.CROP_TYPES[plot['type']]
                    radius = int(self.PLOT_SIZE * 0.4)
                    pos = (rect.centerx, rect.centery)
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, crop_info['color'])
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, crop_info['color'])

        # Draw selector
        sel_rect = pygame.Rect(
            self.GRID_MARGIN_X + self.selector_pos[0] * self.PLOT_SIZE,
            self.GRID_MARGIN_Y + self.selector_pos[1] * self.PLOT_SIZE,
            self.PLOT_SIZE, self.PLOT_SIZE
        )
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = int(155 + 100 * pulse)
        selector_surface = pygame.Surface((self.PLOT_SIZE, self.PLOT_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(selector_surface, (*self.COLOR_SELECTOR, alpha), (0, 0, self.PLOT_SIZE, self.PLOT_SIZE), 3)
        self.screen.blit(selector_surface, sel_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = f"COINS: {self.score}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        time_text = f"TIME: {self.time_remaining // 30:02d}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Harvested crops
        inv_y = self.HEIGHT - 35
        inv_x = 10
        for crop_type, count in self.harvested_crops.items():
            if count > 0:
                crop_info = self.CROP_TYPES[crop_type]
                pygame.draw.circle(self.screen, crop_info['color'], (inv_x + 10, inv_y + 10), 10)
                count_text = f"x{count}"
                count_surf = self.font_main.render(count_text, True, self.COLOR_TEXT)
                self.screen.blit(count_surf, (inv_x + 25, inv_y))
                inv_x += 80

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition_met else "TIME'S UP!"
            color = self.COLOR_GOLD if self.win_condition_met else (200, 50, 50)
            
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, grid_pos, num, color, size_range):
        px, py = (
            self.GRID_MARGIN_X + grid_pos[0] * self.PLOT_SIZE + self.PLOT_SIZE // 2,
            self.GRID_MARGIN_Y + grid_pos[1] * self.PLOT_SIZE + self.PLOT_SIZE // 2
        )
        for _ in range(num):
            self.particles.append(Particle(px, py, color, 30, (-2, 2), (-3, -1), size_range))

    def _create_coin_particles(self, num):
        for _ in range(num):
            px = self.np_random.integers(10, self.WIDTH - 10)
            py = self.HEIGHT - 40
            self.particles.append(Particle(px, py, self.COLOR_GOLD, 40, (-1, 1), (-4, -2), (3, 6)))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "coins_to_win": max(0, self.WIN_SCORE - self.score),
        }
        
    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    
    # --- To run with manual control ---
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Farming Frenzy")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    clock = pygame.time.Clock()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        action[0] = 0 # Default no movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    # Keep displaying the final screen for a moment
    pygame.time.wait(2000)
    env.close()
    print(f"Game Over! Final Score: {info['score']}")