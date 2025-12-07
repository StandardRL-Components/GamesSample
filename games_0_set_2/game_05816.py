
# Generated: 2025-08-28T06:11:16.699933
# Source Brief: brief_05816.md
# Brief Index: 5816

        
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
        "Controls: Arrow keys to move cursor. Space to plant or harvest. Shift to cycle selected crop."
    )

    game_description = (
        "Manage a small isometric farm. Plant, grow, and sell crops to reach 1000 coins before the timer runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 4
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 2700  # 90 seconds at 30fps
        self.MOVE_COOLDOWN_FRAMES = 4

        # Colors
        self.COLOR_BG = (50, 60, 70)
        self.COLOR_UI_BG = (30, 40, 50, 200)
        self.COLOR_TEXT = (230, 240, 250)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)
        self.COLOR_PLOT_PLOWED = (139, 69, 19)
        self.COLOR_PLOT_GROWING = (34, 139, 34)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_MARKET = (160, 82, 45)

        # Crop Data: {id: {time, value, color, ready_color}}
        self.CROP_DATA = {
            1: {'time': 210, 'value': 10, 'color': (100, 220, 100), 'ready_color': (150, 255, 150)}, # 7s
            2: {'time': 360, 'value': 20, 'color': (220, 220, 100), 'ready_color': (255, 255, 150)}, # 12s
            3: {'time': 510, 'value': 35, 'color': (220, 100, 220), 'ready_color': (255, 150, 255)}  # 17s
        }

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Isometric projection helpers
        self.TILE_WIDTH_HALF = 40
        self.TILE_HEIGHT_HALF = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 140

        # State variables (initialized in reset)
        self.farm_plots = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.selected_crop = 1
        self.game_over = False
        self.win_state = None # 'win' or 'lose'
        self.last_space_held = False
        self.last_shift_held = False
        self.move_cooldown = 0
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.win_state = None
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_crop = 1
        
        self.farm_plots = []
        for y in range(self.GRID_SIZE):
            row = []
            for x in range(self.GRID_SIZE):
                row.append({
                    'state': 'plowed', # plowed, growing, ready
                    'crop_type': 0,
                    'growth_timer': 0
                })
            self.farm_plots.append(row)

        self.last_space_held = False
        self.last_shift_held = False
        self.move_cooldown = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- UPDATE GAME LOGIC ---
        self.steps += 1
        self.time_remaining -= 1
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        # 1. Handle Input
        action_info = self._handle_input(movement, space_held, shift_held)
        if action_info['planted']:
            reward += 0.1
        if action_info['harvested']:
            reward += 1.2 # 1.0 for selling, 0.2 for harvesting

        # 2. Update Farm Plots
        self._update_plots()

        # 3. Update Particles
        self._update_particles()
        
        # 4. Update button states
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- CHECK TERMINATION ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            self.win_state = 'win'
        elif self.time_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win_state = 'lose'

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        action_info = {'planted': False, 'harvested': False}

        # Movement
        if self.move_cooldown <= 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1
            elif movement == 2: dy = 1
            elif movement == 3: dx = -1
            elif movement == 4: dx = 1
            
            if dx != 0 or dy != 0:
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)
                self.move_cooldown = self.MOVE_COOLDOWN_FRAMES

        # Cycle Crop (Shift Press)
        if shift_held and not self.last_shift_held:
            self.selected_crop = (self.selected_crop % len(self.CROP_DATA)) + 1
            # sfx: cycle_crop

        # Action (Space Press)
        if space_held and not self.last_space_held:
            cx, cy = self.cursor_pos
            plot = self.farm_plots[cy][cx]
            
            # Plant action
            if plot['state'] == 'plowed':
                plot['state'] = 'growing'
                plot['crop_type'] = self.selected_crop
                plot['growth_timer'] = self.CROP_DATA[self.selected_crop]['time']
                action_info['planted'] = True
                # sfx: plant_seed
                self._create_particles(self._iso_to_screen(cx, cy), 15, self.COLOR_PLOT_PLOWED)
            
            # Harvest action
            elif plot['state'] == 'ready':
                crop_value = self.CROP_DATA[plot['crop_type']]['value']
                self.score += crop_value
                action_info['harvested'] = True
                plot['state'] = 'plowed'
                plot['crop_type'] = 0
                plot['growth_timer'] = 0
                # sfx: harvest_coin
                self._create_particles(self._iso_to_screen(cx, cy), 20, (255, 223, 0), is_coin=True)
        
        return action_info

    def _update_plots(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                plot = self.farm_plots[y][x]
                if plot['state'] == 'growing':
                    plot['growth_timer'] -= 1
                    if plot['growth_timer'] <= 0:
                        plot['state'] = 'ready'
                        # sfx: crop_ready_ding
                        screen_pos = self._iso_to_screen(x, y)
                        self._create_particles(screen_pos, 10, (255, 255, 255, 150), is_burst=True)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += p['grav']
            p['life'] -= 1

    def _create_particles(self, pos, count, color, is_coin=False, is_burst=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(20, 40)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            grav = 0.1
            if is_coin:
                vy = self.np_random.uniform(-4, -2)
                vx = self.np_random.uniform(-1, 1)
                life = self.np_random.integers(40, 60)
            if is_burst:
                vy *= 0.5
            self.particles.append({'x': pos[0], 'y': pos[1], 'vx': vx, 'vy': vy, 'life': life, 'max_life': life, 'color': color, 'grav': grav})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Market Stall
        market_pos = self._iso_to_screen(-1, 0)
        market_points = [
            (market_pos[0], market_pos[1]),
            (market_pos[0] + self.TILE_WIDTH_HALF, market_pos[1] - self.TILE_HEIGHT_HALF),
            (market_pos[0] + self.TILE_WIDTH_HALF, market_pos[1] - self.TILE_HEIGHT_HALF - 40),
            (market_pos[0], market_pos[1] - 40),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_MARKET, market_points)
        roof_points = [
            (market_pos[0] - 5, market_pos[1] - 38),
            (market_pos[0] + self.TILE_WIDTH_HALF + 5, market_pos[1] - self.TILE_HEIGHT_HALF - 38),
            (market_pos[0] + self.TILE_WIDTH_HALF, market_pos[1] - self.TILE_HEIGHT_HALF - 55),
            (market_pos[0], market_pos[1] - 55),
        ]
        pygame.draw.polygon(self.screen, (200, 50, 50), roof_points)


        # Draw plots and crops (back to front)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                plot = self.farm_plots[y][x]
                screen_pos = self._iso_to_screen(x, y)
                
                # Draw plot tile
                plot_color = self.COLOR_PLOT_GROWING if plot['state'] != 'plowed' else self.COLOR_PLOT_PLOWED
                self._draw_iso_tile(screen_pos, plot_color)
                
                # Draw crop
                if plot['state'] in ['growing', 'ready']:
                    self._draw_crop(screen_pos, plot)

        # Draw cursor
        cursor_screen_pos = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        self._draw_iso_tile(cursor_screen_pos, (*self.COLOR_CURSOR, 60), is_highlight=True)
        self._draw_iso_tile(cursor_screen_pos, self.COLOR_CURSOR, is_border=True)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            if len(color) == 4:
                final_color = (*color[:3], alpha)
            else:
                final_color = (*color, alpha)
            
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, final_color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['x'] - 2), int(p['y'] - 2)))


    def _draw_iso_tile(self, pos, color, is_border=False, is_highlight=False):
        points = [
            (pos[0], pos[1]),
            (pos[0] + self.TILE_WIDTH_HALF, pos[1] + self.TILE_HEIGHT_HALF),
            (pos[0], pos[1] + 2 * self.TILE_HEIGHT_HALF),
            (pos[0] - self.TILE_WIDTH_HALF, pos[1] + self.TILE_HEIGHT_HALF),
        ]
        if is_border:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, [(p[0]+1, p[1]) for p in points], color) # Thicken
        elif is_highlight:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, (0,0,0,50))

    def _draw_crop(self, pos, plot):
        crop_data = self.CROP_DATA[plot['crop_type']]
        center_pos = (pos[0], pos[1] + self.TILE_HEIGHT_HALF)
        
        if plot['state'] == 'growing':
            progress = 1.0 - (plot['growth_timer'] / crop_data['time'])
            size = int(progress * 12)
            pygame.draw.circle(self.screen, crop_data['color'], center_pos, max(1, size))
        elif plot['state'] == 'ready':
            size = 14
            # Pulsating glow effect
            glow_size = size + 2 + 2 * math.sin(self.steps * 0.2)
            glow_color = (*crop_data['ready_color'], 50)
            
            temp_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_size, glow_size), glow_size)
            self.screen.blit(temp_surf, (center_pos[0] - glow_size, center_pos[1] - glow_size))

            pygame.draw.circle(self.screen, crop_data['ready_color'], center_pos, size)
            pygame.draw.circle(self.screen, (255,255,255), center_pos, int(size*0.6))


    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Score
        score_text = self.font_m.render(f"💰 {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Timer
        time_str = f"⏰ {int(self.time_remaining / 30) if self.time_remaining > 0 else 0}"
        timer_text = self.font_m.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 8))

        # Selected Crop UI
        crop_ui_surf = pygame.Surface((120, 40), pygame.SRCALPHA)
        crop_ui_surf.fill(self.COLOR_UI_BG)
        
        crop_data = self.CROP_DATA[self.selected_crop]
        pygame.draw.circle(crop_ui_surf, crop_data['color'], (20, 20), 12)
        pygame.draw.circle(crop_ui_surf, (255,255,255,50), (20, 20), 14, 2)
        
        crop_text = self.font_s.render(f"Planting", True, self.COLOR_TEXT)
        crop_ui_surf.blit(crop_text, (40, 12))
        
        self.screen.blit(crop_ui_surf, (self.WIDTH//2 - 60, self.HEIGHT - 40))


        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win_state == 'win':
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_LOSE
            
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "selected_crop": self.selected_crop,
            "cursor_pos": self.cursor_pos
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install pygame
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Farming Sim")
    
    done = False
    total_reward = 0
    
    print(env.user_guide)

    while not done:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()