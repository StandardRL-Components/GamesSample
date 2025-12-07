
# Generated: 2025-08-27T19:08:35.082007
# Source Brief: brief_02054.md
# Brief Index: 2054

        
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
        "Controls: Arrows to move cursor. Space to plant/harvest. "
        "Shift to cycle crops, or sell at the red barn."
    )

    game_description = (
        "Fast-paced farming fun! Plant, grow, and harvest crops to earn $1000 "
        "before time runs out across three stages."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.CELL_WIDTH = self.WIDTH // self.GRID_COLS
        self.CELL_HEIGHT = self.HEIGHT // self.GRID_ROWS
        self.WIN_SCORE = 1000
        self.MAX_STAGES = 3
        self.STAGE_DURATION_SECONDS = 60
        self.STAGE_DURATION_FRAMES = self.STAGE_DURATION_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG = (50, 60, 50)
        self.COLOR_PLOT = (101, 67, 33)
        self.COLOR_BARN = (180, 40, 40)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_INVALID = (255, 0, 0, 150)
        self.COLOR_HARVESTED = (255, 223, 0)
        
        # --- Crop Data ---
        self.CROP_DATA = {
            1: {'name': 'Carrot', 'growth_time': 3 * self.FPS, 'value': 20, 'color': (255, 140, 0)},
            2: {'name': 'Cabbage', 'growth_time': 6 * self.FPS, 'value': 50, 'color': (124, 252, 0)},
            3: {'name': 'Pumpkin', 'growth_time': 10 * self.FPS, 'value': 100, 'color': (255, 100, 20)},
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_pop = pygame.font.SysFont("Arial", 18, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.money = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.time_remaining = 0
        self.plots = []
        self.cursor_pos = [0, 0]
        self.selected_crop_type = 1
        self.available_crops = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []
        self.floating_texts = []
        self.invalid_action_timer = 0
        
        self.barn_pos = (self.GRID_COLS - 1, self.GRID_ROWS - 1)

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.money = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.time_remaining = self.STAGE_DURATION_FRAMES
        
        self.plots = [[self._create_empty_plot() for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
        self.plots[self.barn_pos[0]][self.barn_pos[1]]['state'] = 'barn'
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_crop_type = 1
        self._update_available_crops()
        
        self.prev_space_held = True # Prevent actions on first frame
        self.prev_shift_held = True

        self.particles = []
        self.floating_texts = []
        self.invalid_action_timer = 0
        
        return self._get_observation(), self._get_info()

    def _create_empty_plot(self):
        return {'state': 'empty', 'type': None, 'progress': 0}

    def _start_next_stage(self):
        self.stage += 1
        if self.stage > self.MAX_STAGES:
            self.game_over = True
            self.win = False
            return -100.0 # Terminal penalty
        
        # Reset stage-specific state
        self.time_remaining = self.STAGE_DURATION_FRAMES
        self.plots = [[self._create_empty_plot() for _ in range(self.GRID_ROWS)] for _ in range(self.GRID_COLS)]
        self.plots[self.barn_pos[0]][self.barn_pos[1]]['state'] = 'barn'
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self._update_available_crops()
        self.selected_crop_type = self.available_crops[0]
        
        self.floating_texts.append({
            'text': f"Stage {self.stage}", 'pos': [self.WIDTH // 2, self.HEIGHT // 2],
            'life': self.FPS * 2, 'color': self.COLOR_TEXT, 'vel': [0, -0.5]
        })
        # sfx: stage_start_sound
        return 0.0

    def _update_available_crops(self):
        self.available_crops = [crop for crop in self.CROP_DATA if crop <= self.stage]

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            space_pressed = space_held and not self.prev_space_held
            shift_pressed = shift_held and not self.prev_shift_held
            
            self.prev_space_held, self.prev_shift_held = space_held, shift_held
            
            # --- Update game logic ---
            self.steps += 1
            self.time_remaining -= 1
            if self.invalid_action_timer > 0: self.invalid_action_timer -= 1
            
            reward += self._handle_input(movement, space_pressed, shift_pressed)
            self._update_plots()
            self._update_effects()
            
            # --- Check for game over conditions ---
            terminal_reward = self._check_game_over()
            reward += terminal_reward
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        reward = 0
        
        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

        cx, cy = self.cursor_pos
        plot = self.plots[cx][cy]
        
        # --- Shift Actions ---
        if shift_pressed:
            if plot['state'] == 'barn':
                # Sell crops
                sell_value = 0
                for x in range(self.GRID_COLS):
                    for y in range(self.GRID_ROWS):
                        if self.plots[x][y]['state'] == 'harvested':
                            sell_value += self.CROP_DATA[self.plots[x][y]['type']]['value']
                            self.plots[x][y] = self._create_empty_plot()
                if sell_value > 0:
                    self.money += sell_value
                    reward += 1.0 + (sell_value / 100.0)
                    self._create_particles(cx, cy, 30, self.COLOR_HARVESTED)
                    self.floating_texts.append({
                        'text': f"+${sell_value}", 'pos': [cx * self.CELL_WIDTH, cy * self.CELL_HEIGHT],
                        'life': self.FPS, 'color': self.COLOR_HARVESTED, 'vel': [0, -1]
                    })
                    # sfx: cash_register_sound
                else:
                    self.invalid_action_timer = 10 # sfx: error_buzz
            else:
                # Cycle selected crop
                current_idx = self.available_crops.index(self.selected_crop_type)
                next_idx = (current_idx + 1) % len(self.available_crops)
                self.selected_crop_type = self.available_crops[next_idx]
                # sfx: crop_select_blip
        
        # --- Space Actions ---
        if space_pressed:
            if plot['state'] == 'empty':
                # Plant crop
                plot['state'] = 'growing'
                plot['type'] = self.selected_crop_type
                plot['progress'] = 0
                reward += 0.01
                self._create_particles(cx, cy, 10, self.CROP_DATA[self.selected_crop_type]['color'])
                # sfx: plant_seed_sound
            elif plot['state'] == 'mature':
                # Harvest crop
                plot['state'] = 'harvested'
                plot['progress'] = 0
                reward += 0.02
                self._create_particles(cx, cy, 15, self.COLOR_HARVESTED)
                # sfx: harvest_pop_sound
            else:
                self.invalid_action_timer = 10 # sfx: error_buzz
                
        return reward
        
    def _update_plots(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                plot = self.plots[x][y]
                if plot['state'] == 'growing':
                    plot['progress'] += 1
                    if plot['progress'] >= self.CROP_DATA[plot['type']]['growth_time']:
                        plot['state'] = 'mature'
                        # sfx: crop_ready_chime
    
    def _update_effects(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        self.floating_texts = [t for t in self.floating_texts if t['life'] > 0]
        for t in self.floating_texts:
            t['pos'][0] += t['vel'][0]
            t['pos'][1] += t['vel'][1]
            t['life'] -= 1

    def _check_game_over(self):
        if self.money >= self.WIN_SCORE:
            if not self.game_over:
                self.game_over = True
                self.win = True
                # sfx: win_fanfare
                return 100.0
        
        if self.time_remaining <= 0:
            return self._start_next_stage()

        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw plots ---
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                plot = self.plots[x][y]
                
                if plot['state'] == 'barn':
                    pygame.draw.rect(self.screen, self.COLOR_BARN, rect)
                    pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT, rect)
                    pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

                center_x, center_y = rect.center
                max_radius = min(self.CELL_WIDTH, self.CELL_HEIGHT) // 2 - 5

                if plot['state'] == 'growing':
                    data = self.CROP_DATA[plot['type']]
                    growth_ratio = plot['progress'] / data['growth_time']
                    radius = int(max_radius * growth_ratio)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, data['color'])
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, data['color'])
                elif plot['state'] == 'mature':
                    data = self.CROP_DATA[plot['type']]
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, max_radius, data['color'])
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, max_radius, data['color'])
                    # Pulsing outline to show it's ready
                    pulse = abs(math.sin(self.steps * 0.2))
                    pulse_color = (255, 255, 255, int(255 * pulse))
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, max_radius + 2, pulse_color)
                elif plot['state'] == 'harvested':
                    # Draw a pile of circles
                    for i in range(5):
                        offset_x = self.np_random.integers(-5, 6)
                        offset_y = self.np_random.integers(-5, 6)
                        pygame.gfxdraw.filled_circle(self.screen, center_x + offset_x, center_y + offset_y, max_radius // 2, self.COLOR_HARVESTED)
                        pygame.gfxdraw.aacircle(self.screen, center_x + offset_x, center_y + offset_y, max_radius // 2, self.COLOR_HARVESTED)

        # --- Draw effects ---
        self._render_effects()

        # --- Draw cursor ---
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.CELL_WIDTH, cy * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        cursor_surface = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
        color = self.COLOR_INVALID if self.invalid_action_timer > 0 else self.COLOR_CURSOR
        pygame.draw.rect(cursor_surface, color, cursor_surface.get_rect())
        pygame.draw.rect(cursor_surface, self.COLOR_TEXT, cursor_surface.get_rect(), 3)
        self.screen.blit(cursor_surface, cursor_rect.topleft)
        
    def _render_effects(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['life'] / 3))

        for t in self.floating_texts:
            alpha = min(255, int(255 * (t['life'] / (self.FPS/2))))
            text_surf = self.font_pop.render(t['text'], True, t['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(t['pos'][0]), int(t['pos'][1])))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Money ---
        money_text = f"${self.money} / ${self.WIN_SCORE}"
        self._draw_ui_text(money_text, (10, 5))

        # --- Time ---
        time_str = f"Time: {self.time_remaining // self.FPS:02d}"
        self._draw_ui_text(time_str, (self.WIDTH - 120, 5), right_align=True)
        
        # --- Stage ---
        stage_str = f"Stage: {self.stage}/{self.MAX_STAGES}"
        self._draw_ui_text(stage_str, (self.WIDTH - 120, 30), right_align=True)

        # --- Selected Crop ---
        crop_data = self.CROP_DATA[self.selected_crop_type]
        crop_text = f"Planting: {crop_data['name']}"
        self._draw_ui_text(crop_text, (10, self.HEIGHT - 30))
        
        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.win else "TIME'S UP!"
            end_surf = self.font_ui.render(end_text, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_surf, end_rect)
            
            score_text = f"Final Money: ${self.money}"
            score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(score_surf, score_rect)

    def _draw_ui_text(self, text, pos, right_align=False):
        text_surf = self.font_ui.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect()
        
        padding = 5
        bg_rect = pygame.Rect(
            pos[0] - padding, pos[1] - padding,
            text_rect.width + padding * 2, text_rect.height + padding * 2
        )
        if right_align:
            bg_rect.right = pos[0] + text_rect.width + padding
        
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(bg_surf, bg_rect.topleft)

        if right_align:
            text_rect.topright = (pos[0] + text_rect.width, pos[1])
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, count, color):
        center_x = (grid_x + 0.5) * self.CELL_WIDTH
        center_y = (grid_y + 0.5) * self.CELL_HEIGHT
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color
            })
            
    def _get_info(self):
        return {
            "score": self.money,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining": self.time_remaining // self.FPS,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # For this to work, you need to display the screen.
    # The main class is designed for headless operation.
    
    # Monkey-patch the environment for human play
    class HumanGameEnv(GameEnv):
        def __init__(self):
            super().__init__()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Farming Simulator")

        def _get_observation(self):
            # In human mode, we render directly to the display screen
            super()._get_observation() # Renders to the off-screen surface
            pygame.display.flip() # Update the full display Surface to the screen
            return super()._get_observation() # Return the numpy array as required

    env = HumanGameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Map pygame keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            done = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
    pygame.quit()