
# Generated: 2025-08-28T05:55:29.505128
# Source Brief: brief_05730.md
# Brief Index: 5730

        
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
        "Controls: Use arrow keys to move the selected crystal. "
        "Press Space to select the next crystal and Shift to select the previous one."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Move crystals to match the target pattern before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Game constants
        self._setup_constants()
        self._setup_level_data()

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def _setup_constants(self):
        self.GRID_SIZE = 8
        self.TILE_WIDTH_HALF = 32
        self.TILE_HEIGHT_HALF = 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 140

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (45, 50, 55)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_UI_VALUE = (255, 255, 255)
        self.COLOR_UI_ACCENT = (100, 180, 255)
        self.CRYSTAL_COLORS = {
            'RED': (255, 80, 80),
            'GREEN': (80, 255, 80),
            'BLUE': (80, 120, 255),
            'YELLOW': (255, 255, 80),
            'PURPLE': (180, 80, 255),
            'CYAN': (80, 220, 220),
            'ORANGE': (255, 160, 80)
        }
        self.COLOR_SELECTION = (255, 255, 255)
        self.MAX_STEPS = 2000

    def _setup_level_data(self):
        self.LEVELS = [
            {
                'moves': 20, 'crystals': [
                    {'id': 0, 'color': 'RED', 'start': (2, 3), 'target': (4, 3)},
                    {'id': 1, 'color': 'GREEN', 'start': (3, 2), 'target': (4, 4)},
                    {'id': 2, 'color': 'BLUE', 'start': (2, 2), 'target': (5, 4)},
                ]
            },
            {
                'moves': 35, 'crystals': [
                    {'id': 0, 'color': 'RED', 'start': (1, 1), 'target': (3, 5)},
                    {'id': 1, 'color': 'GREEN', 'start': (2, 1), 'target': (4, 5)},
                    {'id': 2, 'color': 'BLUE', 'start': (5, 2), 'target': (5, 3)},
                    {'id': 3, 'color': 'YELLOW', 'start': (6, 6), 'target': (5, 4)},
                ]
            },
            {
                'moves': 50, 'crystals': [
                    {'id': 0, 'color': 'RED', 'start': (1, 3), 'target': (3, 3)},
                    {'id': 1, 'color': 'GREEN', 'start': (2, 3), 'target': (4, 3)},
                    {'id': 2, 'color': 'BLUE', 'start': (5, 3), 'target': (3, 4)},
                    {'id': 3, 'color': 'YELLOW', 'start': (6, 3), 'target': (4, 4)},
                    {'id': 4, 'color': 'PURPLE', 'start': (3, 6), 'target': (3, 5)},
                ]
            },
            {
                'moves': 70, 'crystals': [
                    {'id': 0, 'color': 'RED', 'start': (1, 1), 'target': (2, 5)},
                    {'id': 1, 'color': 'GREEN', 'start': (2, 1), 'target': (3, 5)},
                    {'id': 2, 'color': 'BLUE', 'start': (3, 1), 'target': (4, 5)},
                    {'id': 3, 'color': 'YELLOW', 'start': (5, 1), 'target': (5, 5)},
                    {'id': 4, 'color': 'PURPLE', 'start': (6, 1), 'target': (6, 5)},
                    {'id': 5, 'color': 'CYAN', 'start': (1, 6), 'target': (1, 2)},
                ]
            },
            {
                'moves': 90, 'crystals': [
                    {'id': 0, 'color': 'RED', 'start': (1, 1), 'target': (6, 1)},
                    {'id': 1, 'color': 'GREEN', 'start': (1, 2), 'target': (6, 2)},
                    {'id': 2, 'color': 'BLUE', 'start': (1, 3), 'target': (6, 3)},
                    {'id': 3, 'color': 'YELLOW', 'start': (6, 6), 'target': (1, 6)},
                    {'id': 4, 'color': 'PURPLE', 'start': (6, 5), 'target': (1, 5)},
                    {'id': 5, 'color': 'CYAN', 'start': (6, 4), 'target': (1, 4)},
                    {'id': 6, 'color': 'ORANGE', 'start': (3, 3), 'target': (3, 0)},
                ]
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.previous_action = [0, 0, 0]
        self.particles = []
        self._load_level(self.level)
        return self._get_observation(), self._get_info()

    def _load_level(self, level_num):
        level_data = self.LEVELS[level_num - 1]
        self.moves_left = level_data['moves']
        self.initial_moves = level_data['moves']
        self.crystals = []
        for c_data in level_data['crystals']:
            self.crystals.append({
                'id': c_data['id'],
                'color': self.CRYSTAL_COLORS[c_data['color']],
                'pos': c_data['start'],
                'target_pos': c_data['target'],
            })
        self.selected_crystal_idx = 0
        self.occupied_pos = {c['pos'] for c in self.crystals}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prev_space, prev_shift = self.previous_action[1] == 1, self.previous_action[2] == 1

        # Handle crystal selection
        if space_held and not prev_space:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.crystals)
            # SFX: UI_SWITCH
        if shift_held and not prev_shift:
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + len(self.crystals)) % len(self.crystals)
            # SFX: UI_SWITCH

        # Handle crystal movement
        if movement != 0:
            crystal = self.crystals[self.selected_crystal_idx]
            old_pos = crystal['pos']
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

            is_valid_move = (0 <= new_pos[0] < self.GRID_SIZE and
                             0 <= new_pos[1] < self.GRID_SIZE and
                             new_pos not in self.occupied_pos)

            if is_valid_move:
                self.moves_left -= 1
                # SFX: CRYSTAL_MOVE
                
                old_dist = abs(old_pos[0] - crystal['target_pos'][0]) + abs(old_pos[1] - crystal['target_pos'][1])
                new_dist = abs(new_pos[0] - crystal['target_pos'][0]) + abs(new_pos[1] - crystal['target_pos'][1])
                
                reward += (old_dist - new_dist)  # +1 if closer, -1 if further

                was_on_target = old_pos == crystal['target_pos']
                is_on_target = new_pos == crystal['target_pos']

                if is_on_target and not was_on_target:
                    reward += 10
                    self._spawn_particles(crystal['pos'], crystal['color'], 20, 'lock') # spawn at old pos
                    # SFX: CRYSTAL_LOCK
                elif was_on_target and not is_on_target:
                    reward -= 10

                self.occupied_pos.remove(old_pos)
                crystal['pos'] = new_pos
                self.occupied_pos.add(new_pos)
            else:
                # SFX: CRYSTAL_BUMP
                pass

        self.previous_action = action

        # Check for level completion
        if all(c['pos'] == c['target_pos'] for c in self.crystals):
            reward += 50
            # SFX: LEVEL_COMPLETE
            self._spawn_particles((self.GRID_SIZE/2, self.GRID_SIZE/2), self.COLOR_UI_ACCENT, 100, 'burst')

            if self.level < len(self.LEVELS):
                self.level += 1
                self._load_level(self.level)
            else:
                self.win = True
                reward += 50  # Extra +50 for winning the game
                self.game_over = True
                terminated = True
                # SFX: GAME_WIN

        # Check for termination conditions
        if self.moves_left <= 0 and not self.game_over:
            reward = -100
            self.game_over = True
            terminated = True
            # SFX: GAME_OVER_LOSE

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _spawn_particles(self, grid_pos, color, count, p_type):
        screen_pos = self._iso_to_screen(*grid_pos)
        for _ in range(count):
            if p_type == 'lock':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = random.randint(15, 30)
            elif p_type == 'burst':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(3, 8)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = random.randint(30, 60)

            self.particles.append({'pos': list(screen_pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                # Fade out
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                radius = int(3 * (p['life'] / p['max_life']))
                if radius > 0:
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_SIZE, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw targets
        for crystal in self.crystals:
            screen_pos = self._iso_to_screen(*crystal['target_pos'])
            points = [
                (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF),
                (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1]),
                (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF),
                (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, (*crystal['color'], 80))

        # Draw crystals
        for i, crystal in enumerate(self.crystals):
            screen_pos = self._iso_to_screen(*crystal['pos'])
            
            # Selection highlight
            if i == self.selected_crystal_idx and not self.game_over:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = int(self.TILE_WIDTH_HALF * (1.1 + pulse * 0.2))
                alpha = int(100 + pulse * 100)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.COLOR_SELECTION, alpha), (radius, radius), radius)
                self.screen.blit(temp_surf, (screen_pos[0] - radius, screen_pos[1] - radius))


            # Crystal shape
            top = (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF)
            right = (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1])
            bottom = (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF)
            left = (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1])
            
            # Draw translucent filled polygon
            points = [top, right, bottom, left]
            pygame.gfxdraw.filled_polygon(self.screen, points, (*crystal['color'], 180))
            pygame.gfxdraw.aapolygon(self.screen, points, crystal['color'])

    def _render_ui(self):
        # Moves left
        moves_text = self.font_medium.render("Moves", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))
        moves_val = self.font_big.render(str(self.moves_left), True, self.COLOR_UI_VALUE)
        self.screen.blit(moves_val, (20, 45))
        
        # Level
        level_text = self.font_medium.render("Level", True, self.COLOR_UI_TEXT)
        tr = level_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(level_text, tr)
        level_val = self.font_big.render(f"{self.level}/{len(self.LEVELS)}", True, self.COLOR_UI_VALUE)
        tr = level_val.get_rect(topright=(self.WIDTH - 20, 45))
        self.screen.blit(level_val, tr)

        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, self.HEIGHT - 30))

        # Target pattern preview
        target_text = self.font_medium.render("Target", True, self.COLOR_UI_TEXT)
        tc = target_text.get_rect(centerx=self.WIDTH // 2, y=10)
        self.screen.blit(target_text, tc)

        preview_origin_x = self.WIDTH // 2
        preview_origin_y = 65
        preview_scale = 0.25
        for crystal in self.crystals:
            px, py = crystal['target_pos']
            sx = preview_origin_x + (px - py) * self.TILE_WIDTH_HALF * preview_scale
            sy = preview_origin_y + (px + py) * self.TILE_HEIGHT_HALF * preview_scale
            pygame.draw.circle(self.screen, crystal['color'], (int(sx), int(sy)), int(self.TILE_WIDTH_HALF * preview_scale))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            end_text = self.font_big.render(msg, True, color)
            tr = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, tr)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually, you would need a different setup that maps keyboard events to actions.
    # This example just runs a few random steps.
    
    obs, info = env.reset()
    print("Initial Info:", info)
    
    terminated = False
    total_reward = 0
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Crystal Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # This is a simple keyboard mapping for manual play
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Info: {info}, Total Reward: {total_reward}")
            pygame.time.wait(2000) # Wait 2 seconds before resetting
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(10) # Limit manual play speed
        
    env.close()