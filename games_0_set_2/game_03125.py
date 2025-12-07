# Generated: 2025-08-28T07:04:27.539909
# Source Brief: brief_03125.md
# Brief Index: 3125

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to select plot/market. Press space to interact (plant/harvest/sell). Press shift to cycle crop type."
    )

    game_description = (
        "Manage an isometric farm. Plant, harvest, and sell crops to earn 1000 coins in 5 minutes."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 300
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.WIN_SCORE = 1000

        # Colors
        self.COLOR_BG = (139, 172, 15) # Light Green
        self.COLOR_SOIL = (92, 53, 40)
        self.COLOR_PLOT_BORDER = (72, 41, 30)
        self.COLOR_MARKET = (210, 180, 140)
        self.COLOR_MARKET_ROOF = (180, 40, 40)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_TIMER_WARN = (255, 80, 80)

        # Crop Data
        self.CROPS = {
            "wheat": {"grow_time": 5 * self.FPS, "value": 1, "color": (245, 222, 179)},
            "corn": {"grow_time": 10 * self.FPS, "value": 3, "color": (255, 215, 0)},
            "pumpkin": {"grow_time": 15 * self.FPS, "value": 5, "color": (255, 140, 0)},
        }
        self.CROP_TYPES = list(self.CROPS.keys())
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 40)

        # Isometric projection constants
        self.TILE_WIDTH = 64
        self.TILE_HEIGHT = 32
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 120

        # Game entities
        self.plots = []
        self.market = None
        self.selectables = []

        # State variables
        self.steps = 0
        self.coins = 0
        self.game_over = False
        self.time_remaining = 0
        self.selected_index = 0
        self.selected_plant_type_idx = 0
        self.harvested_crops = {}
        self.particles = []
        self.last_action = np.array([0, 0, 0])
        self.reward_this_step = 0

        # self.reset() is called here, but validation needs a clean slate
        # so we defer calling it until after validation setup.
        # This is a pattern to allow validation to run from __init__.
        # self.validate_implementation()
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.coins = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.selected_index = 0
        self.selected_plant_type_idx = 0
        self.harvested_crops = {crop: 0 for crop in self.CROP_TYPES}
        self.particles = []
        self.last_action = np.array([0, 0, 0])

        self.plots = [
            {"state": "empty", "crop_type": None, "growth": 0.0, "grid_pos": (col, row)}
            for row in range(2) for col in range(3)
        ]
        self.market = {"grid_pos": (-2, 1)}

        self.selectables = self.plots + [self.market]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if not self.game_over:
            self.steps += 1
            self.time_remaining -= 1

            self._handle_input(action)
            self._update_game_state()
            self._update_particles()
        
        # The order of these calls is critical.
        # _check_termination must be called before _calculate_reward
        # so that the win/loss reward is included in the returned reward value.
        terminated = self._check_termination()
        reward = self._calculate_reward()

        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated is always False in this environment
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space, shift = action
        last_movement, last_space, last_shift = self.last_action

        # Movement is on press
        if movement != 0 and movement != last_movement:
            if movement == 3: # Left
                self.selected_index = (self.selected_index - 1) % len(self.selectables)
            elif movement == 4: # Right
                self.selected_index = (self.selected_index + 1) % len(self.selectables)

        # Space is on press (rising edge)
        if space == 1 and last_space == 0:
            self._action_interact()

        # Shift is on press (rising edge)
        if shift == 1 and last_shift == 0:
            self._action_cycle_crop()

    def _action_interact(self):
        selected = self.selectables[self.selected_index]

        # Is it a plot?
        if 'state' in selected:
            plot = selected
            if plot['state'] == 'empty':
                # Plant
                crop_to_plant = self.CROP_TYPES[self.selected_plant_type_idx]
                plot['state'] = 'growing'
                plot['crop_type'] = crop_to_plant
                plot['growth'] = 0.0
                # sfx: plant_pop
                self._spawn_particles(plot['grid_pos'], self.CROPS[crop_to_plant]['color'], 10)
            
            elif plot['state'] == 'ready':
                # Harvest
                crop_type = plot['crop_type']
                self.harvested_crops[crop_type] += 1
                plot['state'] = 'empty'
                plot['crop_type'] = None
                plot['growth'] = 0.0
                self.reward_this_step += 0.1 # Harvest reward
                # sfx: harvest_swoosh
                self._spawn_particles(plot['grid_pos'], self.CROPS[crop_type]['color'], 15, 'up')

        # Is it the market?
        else:
            total_sale = 0
            sold_something = False
            for crop_type, count in self.harvested_crops.items():
                if count > 0:
                    sale_value = count * self.CROPS[crop_type]['value']
                    total_sale += sale_value
                    self._spawn_particles(self.market['grid_pos'], (255, 223, 0), count * 5, 'fly_to_ui')
                    sold_something = True
            
            if sold_something:
                self.coins += total_sale
                self.harvested_crops = {crop: 0 for crop in self.CROP_TYPES}
                self.reward_this_step += 1.0 # Sell reward
                # sfx: cash_register

    def _action_cycle_crop(self):
        self.selected_plant_type_idx = (self.selected_plant_type_idx + 1) % len(self.CROP_TYPES)
        # sfx: ui_tick

    def _update_game_state(self):
        for plot in self.plots:
            if plot['state'] == 'growing':
                grow_time = self.CROPS[plot['crop_type']]['grow_time']
                plot['growth'] += 1.0 / grow_time
                if plot['growth'] >= 1.0:
                    plot['growth'] = 1.0
                    plot['state'] = 'ready'
                    # sfx: crop_ready_ding

    def _calculate_reward(self):
        # Small penalty for passing time
        self.reward_this_step -= 0.01
        return self.reward_this_step
    
    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.coins >= self.WIN_SCORE:
            self.reward_this_step += 100 # Win reward
            self.game_over = True
            return True
        
        if self.time_remaining <= 0:
            self.reward_this_step -= 10 # Lose reward
            self.game_over = True
            return True
            
        return False

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw all selectables, sorted by y-pos for correct layering
        sorted_selectables = sorted(self.selectables, key=lambda s: s['grid_pos'][0] + s['grid_pos'][1])

        for i, item in enumerate(sorted_selectables):
            is_plot = 'state' in item
            if is_plot:
                self._render_plot(item)
            else:
                self._render_market(item)

        # Draw selector
        selected_item = self.selectables[self.selected_index]
        self._render_selector(selected_item['grid_pos'])
        
    def _render_plot(self, plot):
        center_x, center_y = self._iso_to_screen(plot['grid_pos'][0], plot['grid_pos'][1])
        points = [
            (center_x, center_y - self.TILE_HEIGHT / 2),
            (center_x + self.TILE_WIDTH / 2, center_y),
            (center_x, center_y + self.TILE_HEIGHT / 2),
            (center_x - self.TILE_WIDTH / 2, center_y),
        ]
        
        # Draw plot base
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SOIL)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLOT_BORDER)

        # Draw crop
        if plot['crop_type']:
            crop_data = self.CROPS[plot['crop_type']]
            progress = plot['growth']
            
            if plot['state'] == 'growing':
                # Growing animation
                height = int(progress * 20)
                width = int(5 + progress * 10)
                pygame.draw.ellipse(self.screen, (0, 128, 0), (center_x - width/2, center_y - height, width, height))
            
            elif plot['state'] == 'ready':
                # Ready-to-harvest visual
                size = 18
                color = crop_data['color']
                if plot['crop_type'] == 'wheat':
                    pygame.draw.rect(self.screen, color, (center_x - size/2, center_y - size, size, size))
                elif plot['crop_type'] == 'corn':
                    pygame.draw.ellipse(self.screen, color, (center_x - size/3, center_y - size*1.2, size*0.66, size*1.2))
                elif plot['crop_type'] == 'pumpkin':
                    pygame.gfxdraw.filled_circle(self.screen, center_x, int(center_y - size/2), int(size/2), color)

            # Growth bar
            bar_width = 40
            bar_height = 5
            bar_x = center_x - bar_width / 2
            bar_y = center_y + self.TILE_HEIGHT / 2 + 5
            fill_width = int(bar_width * progress)
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (100,255,100), (bar_x, bar_y, fill_width, bar_height))

    def _render_market(self, market):
        gx, gy = market['grid_pos']
        base_x, base_y = self._iso_to_screen(gx, gy)
        
        # Stall base
        h, w = self.TILE_HEIGHT, self.TILE_WIDTH
        base_points = [
            (base_x, base_y - h/2), (base_x + w/2, base_y),
            (base_x, base_y + h/2), (base_x - w/2, base_y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_MARKET)
        pygame.gfxdraw.aapolygon(self.screen, base_points, (0,0,0,50))

        # Stall posts
        z_offset = 40
        pygame.draw.line(self.screen, self.COLOR_PLOT_BORDER, (base_x - w/4, base_y - h/4), (base_x - w/4, base_y - h/4 - z_offset), 4)
        pygame.draw.line(self.screen, self.COLOR_PLOT_BORDER, (base_x + w/4, base_y - h/4), (base_x + w/4, base_y - h/4 - z_offset), 4)
        
        # Roof
        roof_points = [
            (base_x - w/4 - 5, base_y - h/4 - z_offset),
            (base_x + w/4 + 5, base_y - h/4 - z_offset),
            (base_x + w/4 + 5, base_y - h/4 - z_offset - 10),
            (base_x - w/4 - 5, base_y - h/4 - z_offset - 10)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_MARKET_ROOF, roof_points)

        # Harvested crop display
        y_offset = base_y + h/2 + 10
        for i, (crop, count) in enumerate(self.harvested_crops.items()):
            if count > 0:
                color = self.CROPS[crop]['color']
                pygame.gfxdraw.filled_circle(self.screen, int(base_x - 20 + i*30), y_offset, 5, color)
                text = self.font_small.render(f"x{count}", True, self.COLOR_TEXT)
                self.screen.blit(text, (base_x - 10 + i*30, y_offset-8))


    def _render_selector(self, grid_pos):
        center_x, center_y = self._iso_to_screen(grid_pos[0], grid_pos[1])
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = (*self.COLOR_SELECTOR, alpha)
        
        points = [
            (center_x, center_y - self.TILE_HEIGHT / 2 - 3),
            (center_x + self.TILE_WIDTH / 2 + 3, center_y),
            (center_x, center_y + self.TILE_HEIGHT / 2 + 3),
            (center_x - self.TILE_WIDTH / 2 - 3, center_y),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color) # Draw twice for thickness

    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, (0,0,0,100), (0,0,self.WIDTH, 50))

        # Coin display
        coin_text = self.font_large.render(f"{self.coins}", True, self.COLOR_TEXT)
        coin_shadow = self.font_large.render(f"{self.coins}", True, self.COLOR_TEXT_SHADOW)
        pygame.gfxdraw.filled_circle(self.screen, 25, 25, 15, (255, 223, 0))
        pygame.gfxdraw.aacircle(self.screen, 25, 25, 15, (0,0,0,50))
        self.screen.blit(coin_shadow, (52, 12))
        self.screen.blit(coin_text, (50, 10))

        # Timer display
        time_left_sec = self.time_remaining // self.FPS
        minutes = time_left_sec // 60
        seconds = time_left_sec % 60
        timer_color = self.COLOR_TEXT if time_left_sec > 60 else self.COLOR_TIMER_WARN
        time_text = self.font_large.render(f"{minutes:02}:{seconds:02}", True, timer_color)
        time_shadow = self.font_large.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_TEXT_SHADOW)
        text_rect = time_text.get_rect(topright=(self.WIDTH - 12, 12))
        shadow_rect = time_shadow.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_shadow, shadow_rect)
        self.screen.blit(time_text, text_rect)

        # Selected crop display
        crop_name = self.CROP_TYPES[self.selected_plant_type_idx]
        crop_color = self.CROPS[crop_name]['color']
        text = self.font_small.render(f"Planting: {crop_name.capitalize()}", True, self.COLOR_TEXT)
        self.screen.blit(text, (self.WIDTH // 2 - text.get_width()//2, 10))
        pygame.draw.rect(self.screen, crop_color, (self.WIDTH // 2 - 50, 30, 100, 5))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "You Win!" if self.coins >= self.WIN_SCORE else "Time's Up!"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_pos, color, count, p_type='burst'):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(count):
            if p_type == 'burst':
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            elif p_type == 'up':
                vel = [random.uniform(-1, 1), random.uniform(-2, -4)]
            elif p_type == 'fly_to_ui':
                target_x, target_y = 25, 25
                dx, dy = target_x - sx, target_y - sy
                dist = math.hypot(dx, dy)
                if dist == 0: dist = 1
                vel = [dx/dist * 4, dy/dist * 4]
            
            self.particles.append({
                'pos': [sx, sy], 'vel': vel, 'color': color,
                'life': random.randint(20, 40)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['life'] * 0.15))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], size)

    def _get_info(self):
        return {
            "score": self.coins,
            "steps": self.steps,
            "time_remaining": self.time_remaining // self.FPS,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Farm Manager")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # no-op, released, released

    print(env.user_guide)

    while not done:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        
        # Movement
        mov_action = 0
        # A simple rising edge detection for movement
        # to prevent flying through selections.
        # This is a bit different from the agent's view but better for human play.
        new_mov_action = 0
        if keys[pygame.K_LEFT]: new_mov_action = 3
        elif keys[pygame.K_RIGHT]: new_mov_action = 4
        
        # Action is taken on key press, not hold
        if new_mov_action != action[0] and new_mov_action != 0:
            mov_action = new_mov_action
        else:
            mov_action = 0 # No move
        
        # Buttons
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        current_action = np.array([mov_action, space_action, shift_action])
        
        # --- Gym step ---
        # The environment uses the previous action to detect rising edges
        # so we pass the full state of keys, and the env handles the logic.
        step_action = np.array([
            3 if keys[pygame.K_LEFT] else (4 if keys[pygame.K_RIGHT] else 0),
            space_action,
            shift_action
        ])
        obs, reward, terminated, truncated, info = env.step(step_action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            done = True
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()