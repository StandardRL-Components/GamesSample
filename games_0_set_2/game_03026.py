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
        "Controls: Use arrow keys to select a plot. Press Space to plant or harvest. "
        "Hold Shift to sell all harvested crops."
    )

    game_description = (
        "Manage your farm to earn $500 before time runs out. Select plots, plant seeds, "
        "harvest mature crops, and sell them at the market for profit."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1800  # 60 seconds at 30fps -> 1800, but since it's turn based, let's say 600 actions.
    WIN_SCORE = 500
    
    # Colors
    COLOR_BG = (54, 34, 38)
    COLOR_DIRT = (87, 56, 50)
    COLOR_DIRT_DARK = (69, 45, 40)
    COLOR_GRASS = (64, 101, 51)
    COLOR_STALL_ROOF = (181, 80, 70)
    COLOR_STALL_WOOD = (138, 94, 69)
    COLOR_UI_BG = (40, 25, 28, 200)
    COLOR_TEXT = (255, 243, 227)
    COLOR_TEXT_MONEY = (255, 229, 102)
    COLOR_TEXT_WARN = (255, 100, 100)
    COLOR_SELECTOR = (255, 255, 255)

    CROP_DATA = {
        'wheat': {'grow_time': 100, 'sell_price': 5, 'color': (242, 206, 114)},
        'tomato': {'grow_time': 150, 'sell_price': 10, 'color': (220, 56, 56)},
        'corn': {'grow_time': 200, 'sell_price': 20, 'color': (255, 247, 153)},
        'grape': {'grow_time': 250, 'sell_price': 35, 'color': (128, 79, 153)},
    }
    CROP_TYPES = list(CROP_DATA.keys())

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

        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_floating = pygame.font.Font(None, 28)

        self.plot_positions = [
            (self.SCREEN_WIDTH * 0.35, self.SCREEN_HEIGHT * 0.5),
            (self.SCREEN_WIDTH * 0.65, self.SCREEN_HEIGHT * 0.5),
            (self.SCREEN_WIDTH * 0.35, self.SCREEN_HEIGHT * 0.8),
            (self.SCREEN_WIDTH * 0.65, self.SCREEN_HEIGHT * 0.8),
        ]
        self.iso_tile_width = 100
        self.iso_tile_height = 50

        # The reset is called here, which calls _get_observation, which calls _render_game.
        # It needs to be initialized before reset is called.
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_ran_out = False
        self.plots = []
        self.harvested_storage = {}
        self.selected_plot_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.floating_texts = []
        
        self.reset()
        # self.validate_implementation() # Validation can be run separately if needed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_ran_out = False
        
        self.plots = []
        for i in range(4):
            self.plots.append({
                'crop_type': self.CROP_TYPES[i],
                'state': 'empty', # 'empty', 'growing', 'ready'
                'growth': 0.0,
            })

        self.harvested_storage = {crop: 0 for crop in self.CROP_TYPES}
        self.selected_plot_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.floating_texts = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self.steps += 1

        # --- Update Game Logic ---
        self._update_crop_growth()
        self._update_animations()

        # Handle player input (on rising edge of button press)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        self._handle_movement(movement)
        
        if space_pressed:
            reward += self._handle_interaction()
        
        if shift_pressed:
            reward += self._handle_sell_all()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Calculate continuous rewards ---
        empty_plots = sum(1 for plot in self.plots if plot['state'] == 'empty')
        reward -= empty_plots * 0.01

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False # This game is episode-limited by MAX_STEPS, but we use terminated for that.
        if terminated:
            if self.win_condition_met:
                reward += 100 # Win bonus
            elif self.time_ran_out:
                reward -= 10 # Time out penalty
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_crop_growth(self):
        for i, plot in enumerate(self.plots):
            if plot['state'] == 'growing':
                plot['growth'] += 1.0
                if plot['growth'] >= self.CROP_DATA[plot['crop_type']]['grow_time']:
                    plot['state'] = 'ready'
                    plot['growth'] = self.CROP_DATA[plot['crop_type']]['grow_time']
                    # SFX: Crop ready chime
                    self._create_particles(i, self.CROP_DATA[plot['crop_type']]['color'], 5, 2)

    def _update_animations(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
        
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]
        for ft in self.floating_texts:
            ft['y'] -= ft['vy']
            ft['life'] -= 1
            ft['alpha'] = max(0, 255 * (ft['life'] / ft['max_life']))

    def _handle_movement(self, movement):
        current_row, current_col = self.selected_plot_idx // 2, self.selected_plot_idx % 2
        if movement == 1: # Up
            new_row = max(0, current_row - 1)
            self.selected_plot_idx = new_row * 2 + current_col
        elif movement == 2: # Down
            new_row = min(1, current_row + 1)
            self.selected_plot_idx = new_row * 2 + current_col
        elif movement == 3: # Left
            new_col = max(0, current_col - 1)
            self.selected_plot_idx = current_row * 2 + new_col
        elif movement == 4: # Right
            new_col = min(1, current_col + 1)
            self.selected_plot_idx = current_row * 2 + new_col

    def _handle_interaction(self):
        plot = self.plots[self.selected_plot_idx]
        if plot['state'] == 'empty':
            # Plant
            plot['state'] = 'growing'
            plot['growth'] = 0
            # SFX: Plant seed
            self._create_particles(self.selected_plot_idx, self.COLOR_DIRT, 10, 1)
            return 0 # No immediate reward for planting
        elif plot['state'] == 'ready':
            # Harvest
            crop_type = plot['crop_type']
            self.harvested_storage[crop_type] += 1
            plot['state'] = 'empty'
            plot['growth'] = 0
            # SFX: Harvest pop
            self._create_particles(self.selected_plot_idx, self.CROP_DATA[crop_type]['color'], 20, 3)
            return 0.1 # Reward for harvesting
        return 0

    def _handle_sell_all(self):
        money_gained = 0
        for crop_type, count in self.harvested_storage.items():
            if count > 0:
                price = self.CROP_DATA[crop_type]['sell_price']
                money_gained += count * price
                self.harvested_storage[crop_type] = 0
        
        if money_gained > 0:
            self.score += money_gained
            # SFX: Cha-ching
            self._create_floating_text(f"+${money_gained}", self.SCREEN_WIDTH - 80, 50, self.COLOR_TEXT_MONEY)
            return 1.0 # Reward for selling
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_GRASS)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw market stall
        stall_base = (self.SCREEN_WIDTH * 0.85, self.SCREEN_HEIGHT * 0.25)
        pygame.draw.rect(self.screen, self.COLOR_STALL_WOOD, (stall_base[0] - 50, stall_base[1], 100, 60))
        pygame.draw.polygon(self.screen, self.COLOR_STALL_ROOF, [
            (stall_base[0] - 60, stall_base[1]),
            (stall_base[0] + 60, stall_base[1]),
            (stall_base[0], stall_base[1] - 30)
        ])

        # Draw plots
        for i, plot in enumerate(self.plots):
            center_x, center_y = self.plot_positions[i]
            self._draw_iso_tile(self.screen, self.COLOR_DIRT, (center_x, center_y), self.iso_tile_width, self.iso_tile_height, self.COLOR_DIRT_DARK)

            # Draw crop
            if plot['state'] != 'empty':
                crop_data = self.CROP_DATA[plot['crop_type']]
                growth_ratio = plot['growth'] / crop_data['grow_time']
                
                max_radius = self.iso_tile_height * 0.4
                radius = int(max_radius * growth_ratio)
                
                if plot['state'] == 'ready':
                    # Pulsing effect for ready crops
                    pulse = abs(math.sin(self.steps * 0.1))
                    radius = int(max_radius + pulse * 4)
                    pos = (int(center_x), int(center_y - self.iso_tile_height * 0.25))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, crop_data['color'])
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, crop_data['color'])
                    # Shine effect
                    shine_pos = (center_x + radius * 0.4, center_y - self.iso_tile_height * 0.25 - radius * 0.4)
                    shine_surf = pygame.Surface((radius, radius), pygame.SRCALPHA)
                    pygame.draw.circle(shine_surf, (255,255,255,150), (radius*0.7, radius*0.3), int(radius*0.2))
                    self.screen.blit(shine_surf, (shine_pos[0]-radius*0.7, shine_pos[1]-radius*0.3))
                else: # Growing
                    pos = (int(center_x), int(center_y - self.iso_tile_height * 0.25))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, crop_data['color'])
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, crop_data['color'])

        # Draw selector
        sel_x, sel_y = self.plot_positions[self.selected_plot_idx]
        pulse_alpha = 150 + 105 * math.sin(self.steps * 0.2)
        selector_color = (*self.COLOR_SELECTOR, pulse_alpha)
        self._draw_iso_tile(self.screen, (0,0,0,0), (sel_x, sel_y), self.iso_tile_width + 8, self.iso_tile_height + 4, selector_color, 3)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_ui(self):
        # UI Background panels
        ui_bg_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(ui_bg_surf, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, 45))
        pygame.draw.rect(ui_bg_surf, self.COLOR_UI_BG, (0, self.SCREEN_HEIGHT - 65, self.SCREEN_WIDTH, 65))
        self.screen.blit(ui_bg_surf, (0,0))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        timer_text = f"TIME: {time_left:.1f}"
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_TEXT_WARN
        self._draw_text(timer_text, (20, 22), self.font_main, timer_color, "left")

        # Money
        money_text = f"${self.score}"
        self._draw_text(money_text, (self.SCREEN_WIDTH - 20, 22), self.font_main, self.COLOR_TEXT_MONEY, "right")

        # Harvested goods panel
        storage_start_x = self.SCREEN_WIDTH / 2 - (len(self.CROP_TYPES) * 120) / 2
        for i, crop_type in enumerate(self.CROP_TYPES):
            x = storage_start_x + i * 120
            y = self.SCREEN_HEIGHT - 35
            
            crop_color = self.CROP_DATA[crop_type]['color']
            dark_crop_color = tuple(max(0, c * 0.8) for c in crop_color)
            pygame.draw.circle(self.screen, crop_color, (x, y), 15)
            pygame.draw.circle(self.screen, dark_crop_color, (x,y), 15, 2)
            
            count_text = f"x {self.harvested_storage[crop_type]}"
            self._draw_text(count_text, (x + 25, y), self.font_main, self.COLOR_TEXT, "left")

        # Floating texts
        for ft in self.floating_texts:
            text_surf = self.font_floating.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(ft['alpha'])
            self.screen.blit(text_surf, (ft['x'], ft['y']))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition_met:
                end_text = "YOU WIN!"
                end_color = self.COLOR_TEXT_MONEY
            else:
                end_text = "TIME'S UP!"
                end_color = self.COLOR_TEXT_WARN
            
            self._draw_text(end_text, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), pygame.font.Font(None, 72), end_color)
            self._draw_text(f"Final Score: ${self.score}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30), self.font_main, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "left":
            text_rect.midleft = pos
        elif align == "right":
            text_rect.midright = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_iso_tile(self, surface, color, pos, width, height, outline_color=None, outline_width=2):
        x, y = pos
        points = [
            (x, y - height / 2),
            (x + width / 2, y),
            (x, y + height / 2),
            (x - width / 2, y)
        ]
        int_points = [(int(px), int(py)) for px, py in points]
        
        # FIX: Handle both RGB (len 3) and RGBA (len 4) colors. Draw if not fully transparent.
        if len(color) == 3 or (len(color) == 4 and color[3] > 0):
            pygame.gfxdraw.filled_polygon(surface, int_points, color)
        
        if outline_color:
            # aalines doesn't support thickness, so we draw multiple lines for a thicker effect
            # also, aalines doesn't handle alpha well, so we draw to a temp surface
            temp_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            
            for i in range(outline_width):
                # Create slightly offset points for thickness
                p_outer = [(p[0], p[1]-i) for p in int_points]
                p_inner = [(p[0], p[1]+i) for p in int_points]
                p_left = [(p[0]-i, p[1]) for p in int_points]
                p_right = [(p[0]+i, p[1]) for p in int_points]
                pygame.draw.aalines(temp_surf, outline_color, True, int_points, 1)
                pygame.draw.aalines(temp_surf, outline_color, True, p_outer, 1)
                pygame.draw.aalines(temp_surf, outline_color, True, p_inner, 1)
                pygame.draw.aalines(temp_surf, outline_color, True, p_left, 1)
                pygame.draw.aalines(temp_surf, outline_color, True, p_right, 1)

            surface.blit(temp_surf, (0, 0))


    def _create_particles(self, plot_idx, color, count, size):
        x, y = self.plot_positions[plot_idx]
        for _ in range(count):
            self.particles.append({
                'x': x + random.uniform(-10, 10),
                'y': y - self.iso_tile_height * 0.25 + random.uniform(-10, 10),
                'vx': random.uniform(-1, 1),
                'vy': random.uniform(-3, -1),
                'size': random.uniform(size, size * 1.5),
                'color': color,
                'life': random.randint(20, 40)
            })

    def _create_floating_text(self, text, x, y, color):
        self.floating_texts.append({
            'text': text,
            'x': x, 'y': y,
            'vy': 1.5,
            'color': color,
            'life': 60,
            'max_life': 60,
            'alpha': 255
        })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win_condition_met = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.time_ran_out = True
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.MAX_STEPS - self.steps),
            "selected_plot": self.selected_plot_idx,
            "storage": self.harvested_storage,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium's standard `play` utility may not work well with
    # the MultiDiscrete action space and keyboard mapping. This is a simple
    # manual player.
    
    # To run with display, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Setup a visible window for playing
    pygame.display.set_caption("Farming Game")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and terminated:
                # Restart game if terminated
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            # Use get_just_pressed logic for space and shift to match agent behavior
            space_key_down = keys[pygame.K_SPACE]
            shift_key_down = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [movement, 1 if space_key_down else 0, 1 if shift_key_down else 0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(15) # Control the speed of the manual play

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Press any key to restart.")

    env.close()