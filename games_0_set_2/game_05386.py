
# Generated: 2025-08-28T04:51:03.343002
# Source Brief: brief_05386.md
# Brief Index: 5386

        
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

    user_guide = (
        "Controls: Use arrow keys to select a plot. Press Space to plant a seed. "
        "Hold Shift to sell harvested crops."
    )

    game_description = (
        "Manage your isometric farm to earn 500 coins by planting, harvesting, "
        "and selling crops before time runs out."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self._define_constants()
        self._setup_pygame()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        self.reset()
        self.validate_implementation()

    def _define_constants(self):
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # Game parameters
        self.GRID_ROWS = 5
        self.GRID_COLS = 5
        self.MAX_STEPS = 3000
        self.WIN_SCORE = 500
        self.CROP_GROW_TIME = 200
        self.CROP_SELL_PRICE = 5
        self.INITIAL_SEEDS = 10
        
        # Visuals
        self.TILE_WIDTH = 64
        self.TILE_HEIGHT = 32
        self.FARM_CENTER_X = self.SCREEN_WIDTH // 2
        self.FARM_CENTER_Y = self.SCREEN_HEIGHT // 2 - 20

        # Colors
        self.COLOR_BG = (34, 40, 49)
        self.COLOR_SOIL = (94, 65, 47)
        self.COLOR_SOIL_DARK = (69, 47, 34)
        self.COLOR_CROP_STEM = (60, 120, 60)
        self.COLOR_CROP_LEAF = (80, 180, 80)
        self.COLOR_SELECTION = (255, 255, 0)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_TEXT = (238, 238, 238)
        self.COLOR_TIMER_WARN = (255, 100, 100)
        self.COLOR_COIN = (255, 215, 0)
        self.COLOR_WIN = (127, 255, 0)
        self.COLOR_LOSE = (220, 20, 60)

    def _setup_pygame(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_feedback = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_STEPS
        
        self.seeds = self.INITIAL_SEEDS
        self.harvested_crops = 0
        
        self.selected_tile = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        
        self.soil_patches = [
            [{'state': 'empty', 'growth': 0} for _ in range(self.GRID_COLS)]
            for _ in range(self.GRID_ROWS)
        ]
        
        self.particles = []
        self.feedback_text = ""
        self.feedback_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        self.timer -= 1
        
        step_reward = -0.01  # Small penalty for time passing
        
        self._update_crops()
        harvest_reward = self._auto_harvest()
        step_reward += harvest_reward
        
        self._handle_actions(movement, space_pressed, shift_pressed)
        
        sell_reward = 0
        if shift_pressed and self.harvested_crops > 0:
            # sfx: cha-ching!
            coins_earned = self.harvested_crops * self.CROP_SELL_PRICE
            self.score += coins_earned
            self._add_particles(self.harvested_crops, self.COLOR_COIN, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.harvested_crops = 0
            sell_reward = 10.0
            self._set_feedback(f"+{coins_earned} Coins!", self.COLOR_COIN)
        elif shift_pressed:
            self._set_feedback("Nothing to sell!", self.COLOR_TIMER_WARN)

        step_reward += sell_reward
        
        terminated = self.timer <= 0 or self.score >= self.WIN_SCORE
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                self.win = True
                step_reward += 100.0
        
        self._update_feedback()
        self._update_particles()
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _update_crops(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                patch = self.soil_patches[r][c]
                if patch['state'] == 'growing':
                    patch['growth'] += 1
                    if patch['growth'] >= self.CROP_GROW_TIME:
                        patch['state'] = 'ready'
                        # sfx: sparkle/chime
    
    def _auto_harvest(self):
        harvest_reward = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                patch = self.soil_patches[r][c]
                if patch['state'] == 'ready':
                    patch['state'] = 'empty'
                    patch['growth'] = 0
                    self.harvested_crops += 1
                    harvest_reward += 1.0
                    # sfx: pop
                    pos = self._iso_to_screen(r, c)
                    self._add_particles(5, self.COLOR_CROP_LEAF, (pos[0], pos[1] - self.TILE_HEIGHT))
        return harvest_reward

    def _handle_actions(self, movement, space_pressed, shift_pressed):
        # Movement
        if movement == 1: self.selected_tile[0] = max(0, self.selected_tile[0] - 1)
        elif movement == 2: self.selected_tile[0] = min(self.GRID_ROWS - 1, self.selected_tile[0] + 1)
        elif movement == 3: self.selected_tile[1] = max(0, self.selected_tile[1] - 1)
        elif movement == 4: self.selected_tile[1] = min(self.GRID_COLS - 1, self.selected_tile[1] + 1)
        
        # Plant
        if space_pressed:
            r, c = self.selected_tile
            patch = self.soil_patches[r][c]
            if patch['state'] == 'empty':
                if self.seeds > 0:
                    # sfx: plant seed
                    patch['state'] = 'growing'
                    self.seeds -= 1
                    self._set_feedback("Seed planted!", self.COLOR_CROP_LEAF)
                else:
                    # sfx: error buzz
                    self._set_feedback("No seeds left!", self.COLOR_TIMER_WARN)
            else:
                self._set_feedback("Cannot plant here!", self.COLOR_TIMER_WARN)

    def _set_feedback(self, text, color):
        self.feedback_text = text
        self.feedback_color = color
        self.feedback_timer = 60 # frames

    def _update_feedback(self):
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_text = ""

    def _add_particles(self, count, color, pos):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [random.uniform(-1.5, 1.5), random.uniform(-2, -0.5)],
                'life': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_farm()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "seeds": self.seeds,
            "harvested_crops": self.harvested_crops,
        }

    def _iso_to_screen(self, row, col):
        x = self.FARM_CENTER_X + (col - row) * (self.TILE_WIDTH / 2)
        y = self.FARM_CENTER_Y + (col + row) * (self.TILE_HEIGHT / 2)
        return int(x), int(y)

    def _draw_iso_poly(self, surface, color, points, outline_color=None, width=1):
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.draw.polygon(surface, outline_color, points, width)

    def _render_farm(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                patch = self.soil_patches[r][c]
                screen_pos = self._iso_to_screen(r, c)
                
                # Tile points
                top = (screen_pos[0], screen_pos[1])
                left = (screen_pos[0] - self.TILE_WIDTH // 2, screen_pos[1] + self.TILE_HEIGHT // 2)
                bottom = (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT)
                right = (screen_pos[0] + self.TILE_WIDTH // 2, screen_pos[1] + self.TILE_HEIGHT // 2)
                points = [top, left, bottom, right]

                # Draw tile
                self._draw_iso_poly(self.screen, self.COLOR_SOIL, points)
                
                # Draw crop if growing/ready
                if patch['state'] in ['growing', 'ready']:
                    growth_factor = min(1.0, patch['growth'] / self.CROP_GROW_TIME)
                    crop_height = self.TILE_HEIGHT * 1.5 * growth_factor
                    
                    # Stem
                    stem_bottom = (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT)
                    stem_top = (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT - crop_height)
                    pygame.draw.line(self.screen, self.COLOR_CROP_STEM, stem_bottom, stem_top, max(1, int(4 * growth_factor)))
                    
                    # Leaves
                    if growth_factor > 0.2:
                        leaf_size = self.TILE_WIDTH / 3 * growth_factor
                        leaf_y = stem_top[1] + 5
                        leaf_points = [
                            (stem_top[0], leaf_y - leaf_size / 2),
                            (stem_top[0] - leaf_size, leaf_y),
                            (stem_top[0], leaf_y + leaf_size / 2),
                            (stem_top[0] + leaf_size, leaf_y)
                        ]
                        self._draw_iso_poly(self.screen, self.COLOR_CROP_LEAF, leaf_points)

                # Draw selection highlight
                if [r, c] == self.selected_tile:
                    pygame.draw.polygon(self.screen, self.COLOR_SELECTION, points, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))
            
    def _render_ui(self):
        # Top bar
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        # Score
        score_text = self.font_ui.render(f"Coins: {self.score} / {self.WIN_SCORE}", True, self.COLOR_COIN)
        self.screen.blit(score_text, (10, 10))

        # Timer
        timer_color = self.COLOR_TEXT if self.timer > 600 else self.COLOR_TIMER_WARN
        timer_text = self.font_ui.render(f"Time: {self.timer}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Bottom bar for inventory
        bottom_bar = pygame.Surface((self.SCREEN_WIDTH, 30), pygame.SRCALPHA)
        bottom_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(bottom_bar, (0, self.SCREEN_HEIGHT - 30))

        # Seeds
        seeds_text = self.font_ui.render(f"Seeds: {self.seeds}", True, self.COLOR_TEXT)
        self.screen.blit(seeds_text, (10, self.SCREEN_HEIGHT - 25))
        
        # Harvested
        harvest_text = self.font_ui.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_CROP_LEAF)
        self.screen.blit(harvest_text, (self.SCREEN_WIDTH - harvest_text.get_width() - 10, self.SCREEN_HEIGHT - 25))

        # Feedback text
        if self.feedback_text and self.feedback_timer > 0:
            alpha = max(0, min(255, int(255 * (self.feedback_timer / 60))))
            feedback_surf = self.font_feedback.render(self.feedback_text, True, self.feedback_color)
            feedback_surf.set_alpha(alpha)
            pos = (self.SCREEN_WIDTH // 2 - feedback_surf.get_width() // 2, self.SCREEN_HEIGHT - 60)
            self.screen.blit(feedback_surf, pos)

        # Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "TIME'S UP!"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        # Default action is no-op
        action = [0, 0, 0] # move, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if not terminated:
            # Map keyboard to MultiDiscrete action
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4

            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != -0.01: # Don't spam the console with the time penalty
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control the speed of the game for human play

    env.close()