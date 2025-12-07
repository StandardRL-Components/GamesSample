
# Generated: 2025-08-28T06:12:28.469280
# Source Brief: brief_02849.md
# Brief Index: 2849

        
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
        "Controls: ↑↓ to select a floor. ←→ to select a building type. "
        "Space to build. Shift to upgrade the selected floor."
    )

    game_description = (
        "Build and manage a tiny isometric tower to maximize your profits. "
        "Reach $100,000 to win, but don't go bankrupt!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WIN_CONDITION = 100000
    MAX_STEPS = 2000
    STARTING_MONEY = 1500

    # Colors
    COLOR_BG_DAY = (135, 206, 235)
    COLOR_BG_NIGHT = (25, 25, 112)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_UI_BG = (10, 10, 40, 200)
    COLOR_UI_TEXT = (240, 240, 255)
    COLOR_SELECTION = (255, 255, 0)
    COLOR_MONEY_PLUS = (173, 255, 47)
    COLOR_MONEY_MINUS = (255, 69, 0)
    
    FLOOR_TYPES = {
        "FOOD": {"name": "Food", "color": (220, 20, 60), "cost": 500, "income": 20},
        "SERVICE": {"name": "Service", "color": (65, 105, 225), "cost": 2000, "income": 90},
        "CREATIVE": {"name": "Creative", "color": (148, 0, 211), "cost": 8000, "income": 400},
        "RECREATION": {"name": "Recreation", "color": (50, 205, 50), "cost": 25000, "income": 1500},
    }
    FLOOR_TYPE_KEYS = list(FLOOR_TYPES.keys())
    
    UNLOCK_MILESTONES = {
        "SERVICE": 10000,
        "CREATIVE": 25000,
        "RECREATION": 50000,
    }

    # Isometric projection values
    ISO_TILE_WIDTH = 64
    ISO_TILE_HEIGHT = ISO_TILE_WIDTH // 2
    ISO_FLOOR_Z_HEIGHT = ISO_TILE_HEIGHT - 4
    TOWER_ORIGIN_X = SCREEN_WIDTH // 2
    TOWER_ORIGIN_Y = SCREEN_HEIGHT - 50

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
        
        self.font_ui = pygame.font.Font(None, 24)
        self.font_floor = pygame.font.Font(None, 16)
        self.font_particle = pygame.font.Font(None, 18)
        self.font_build_select = pygame.font.Font(None, 28)

        # These are initialized in reset()
        self.money = 0
        self.steps = 0
        self.game_over = False
        self.floors = []
        self.particles = []
        self.selected_floor_idx = -1
        self.selected_build_idx = 0
        self.unlocked_floor_types = []
        self.total_money_earned = 0
        
        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.money = self.STARTING_MONEY
        self.steps = 0
        self.game_over = False
        self.floors = []
        self.particles = []
        self.selected_floor_idx = -1
        self.selected_build_idx = 0
        self.unlocked_floor_types = [self.FLOOR_TYPE_KEYS[0]]
        self.total_money_earned = self.STARTING_MONEY

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- 1. Handle Input ---
        # Select floor to interact with
        if movement == 1:  # Up
            self.selected_floor_idx = min(self.selected_floor_idx + 1, len(self.floors) - 1)
        elif movement == 2:  # Down
            self.selected_floor_idx = max(self.selected_floor_idx - 1, -1) # -1 deselects
        
        # Select floor type to build
        if movement == 3:  # Left
            self.selected_build_idx = (self.selected_build_idx - 1) % len(self.unlocked_floor_types)
        elif movement == 4:  # Right
            self.selected_build_idx = (self.selected_build_idx + 1) % len(self.unlocked_floor_types)
            
        # Build action
        if space_held:
            build_type_key = self.unlocked_floor_types[self.selected_build_idx]
            cost = self.FLOOR_TYPES[build_type_key]["cost"]
            if self.money >= cost:
                self.money -= cost
                new_floor = {
                    "type": build_type_key,
                    "level": 1,
                    "y_pos": self.TOWER_ORIGIN_Y - len(self.floors) * self.ISO_FLOOR_Z_HEIGHT,
                    "build_step": self.steps
                }
                self.floors.append(new_floor)
                self.selected_floor_idx = len(self.floors) - 1
                reward += 1.0
                self._create_text_particle(f"-${cost}", (self.TOWER_ORIGIN_X, new_floor['y_pos']), self.COLOR_MONEY_MINUS)
                # sfx: build_complete.wav
            else:
                pass # sfx: action_fail.wav

        # Upgrade action
        if shift_held and self.selected_floor_idx != -1 and self.selected_floor_idx < len(self.floors):
            floor = self.floors[self.selected_floor_idx]
            difficulty_mod = 1 + (self.total_money_earned // 10000) * 0.1
            upgrade_cost = int(self.FLOOR_TYPES[floor["type"]]["cost"] * (floor["level"] * 0.5) * difficulty_mod)
            if self.money >= upgrade_cost:
                self.money -= upgrade_cost
                floor["level"] += 1
                reward += 2.0
                self._create_text_particle(f"-${upgrade_cost}", (self.TOWER_ORIGIN_X, floor['y_pos']), self.COLOR_MONEY_MINUS)
                # sfx: upgrade_complete.wav
            else:
                pass # sfx: action_fail.wav

        # --- 2. Update Game State ---
        self.steps += 1
        
        # Income generation
        daily_income = 0
        for floor in self.floors:
            floor_info = self.FLOOR_TYPES[floor["type"]]
            income = floor_info["income"] * floor["level"]
            daily_income += income
        
        money_before = self.money
        self.money += daily_income
        money_earned_this_step = self.money - money_before
        
        if money_earned_this_step > 0:
            self.total_money_earned += money_earned_this_step
            reward += (money_earned_this_step / 100) * 0.1
            if self.steps % 5 == 0: # Don't spam particles every step
                self._create_text_particle(f"+${daily_income}", (self.TOWER_ORIGIN_X + 60, self.SCREEN_HEIGHT / 2), self.COLOR_MONEY_PLUS)

        # Unlock new floor types
        for f_type, milestone in self.UNLOCK_MILESTONES.items():
            if self.total_money_earned >= milestone and f_type not in self.unlocked_floor_types:
                self.unlocked_floor_types.append(f_type)
                # sfx: unlock.wav

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["y"] -= p["vy"]
            p["life"] -= 1

        # --- 3. Check Termination ---
        terminated = False
        if self.money >= self.WIN_CONDITION:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.money <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.money,
            "steps": self.steps,
            "floors": len(self.floors),
            "total_earnings": self.total_money_earned
        }

    def _render_all(self):
        self._draw_background()
        self._draw_tower_base()
        self._draw_floors()
        self._draw_particles()
        self._draw_ui()

    def _draw_background(self):
        # Day/night cycle
        cycle_duration = 600 # steps for a full day-night cycle
        time_of_day = (self.steps % cycle_duration) / cycle_duration # 0.0 to 1.0
        
        if time_of_day < 0.5: # Day
            t = time_of_day * 2 # 0 to 1
            bg_color = self._lerp_color(self.COLOR_BG_DAY, self.COLOR_BG_NIGHT, math.sin(t * math.pi / 2)**2)
        else: # Night
            t = (time_of_day - 0.5) * 2 # 0 to 1
            bg_color = self._lerp_color(self.COLOR_BG_NIGHT, self.COLOR_BG_DAY, math.sin(t * math.pi / 2)**2)
            
        self.screen.fill(bg_color)

    def _draw_tower_base(self):
        self._draw_iso_cube(
            (self.TOWER_ORIGIN_X, self.TOWER_ORIGIN_Y), 
            (100, 100, 100), 
            self.ISO_TILE_WIDTH * 1.2, 
            self.ISO_FLOOR_Z_HEIGHT
        )

    def _draw_floors(self):
        for i, floor in enumerate(self.floors):
            floor_info = self.FLOOR_TYPES[floor["type"]]
            is_selected = (i == self.selected_floor_idx)
            
            self._draw_iso_cube(
                (self.TOWER_ORIGIN_X, floor["y_pos"]), 
                floor_info["color"], 
                self.ISO_TILE_WIDTH, 
                self.ISO_FLOOR_Z_HEIGHT,
                is_selected
            )
            
            # Draw floor info text
            text_surf = self.font_floor.render(f"{floor_info['name']} Lvl:{floor['level']}", True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.TOWER_ORIGIN_X, floor["y_pos"] - self.ISO_FLOOR_Z_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_iso_cube(self, pos, color, width, height, is_selected=False):
        x, y = pos
        hw, hh = width / 2, width / 4 # height of diamond is half its width
        
        top_points = [
            (x, y - hh), (x + hw, y), (x, y + hh), (x - hw, y)
        ]
        
        # Darken color for sides
        side_color = tuple(max(0, c - 50) for c in color)
        dark_side_color = tuple(max(0, c - 80) for c in color)

        # Right side face
        pygame.gfxdraw.filled_polygon(self.screen, [
            (x, y + hh), (x + hw, y), (x + hw, y + height), (x, y + hh + height)
        ], side_color)
        
        # Left side face
        pygame.gfxdraw.filled_polygon(self.screen, [
            (x, y + hh), (x - hw, y), (x - hw, y + height), (x, y + hh + height)
        ], dark_side_color)

        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)

        # Outline
        pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_BLACK)
        pygame.gfxdraw.aapolygon(self.screen, [
            (x, y + hh), (x + hw, y), (x + hw, y + height), (x, y + hh + height)
        ], self.COLOR_BLACK)
        pygame.gfxdraw.aapolygon(self.screen, [
            (x, y + hh), (x - hw, y), (x - hw, y + height), (x, y + hh + height)
        ], self.COLOR_BLACK)
        
        if is_selected:
            # Pulsing glow effect
            glow_alpha = 128 + 127 * math.sin(self.steps * 0.2)
            glow_color = (*self.COLOR_SELECTION, glow_alpha)
            
            # Draw thicker outline for selection
            for i in range(4):
                offset_points = [(px+i, py) for px,py in top_points]
                pygame.gfxdraw.aapolygon(self.screen, offset_points, glow_color)
                offset_points = [(px-i, py) for px,py in top_points]
                pygame.gfxdraw.aapolygon(self.screen, offset_points, glow_color)
                offset_points = [(px, py+i) for px,py in top_points]
                pygame.gfxdraw.aapolygon(self.screen, offset_points, glow_color)
                offset_points = [(px, py-i) for px,py in top_points]
                pygame.gfxdraw.aapolygon(self.screen, offset_points, glow_color)


    def _draw_ui(self):
        # Top bar
        ui_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        s = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, 0))

        # Money display
        money_text = f"$ {int(self.money):,}"
        money_surf = self.font_ui.render(money_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(money_surf, (10, 10))
        
        # Day display
        day_text = f"Day: {self.steps}/{self.MAX_STEPS}"
        day_surf = self.font_ui.render(day_text, True, self.COLOR_UI_TEXT)
        day_rect = day_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(day_surf, day_rect)
        
        # Build selection UI
        build_ui_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60)
        s = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (0, self.SCREEN_HEIGHT - 60))

        selected_key = self.unlocked_floor_types[self.selected_build_idx]
        selected_info = self.FLOOR_TYPES[selected_key]
        
        build_text = f"Build: {selected_info['name']} (Cost: ${selected_info['cost']:,})"
        build_surf = self.font_build_select.render(build_text, True, selected_info['color'])
        build_rect = build_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 35))
        self.screen.blit(build_surf, build_rect)

        # Arrows for selection
        left_arrow = self.font_build_select.render("<", True, self.COLOR_UI_TEXT)
        right_arrow = self.font_build_select.render(">", True, self.COLOR_UI_TEXT)
        self.screen.blit(left_arrow, (build_rect.left - 30, build_rect.top))
        self.screen.blit(right_arrow, (build_rect.right + 15, build_rect.top))
        
        # Win/Loss Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0,0))
            
            if self.money >= self.WIN_CONDITION:
                msg = "YOU WIN!"
                color = self.COLOR_MONEY_PLUS
            else:
                msg = "GAME OVER"
                color = self.COLOR_MONEY_MINUS
            
            msg_font = pygame.font.Font(None, 80)
            msg_surf = msg_font.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)


    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["lifespan"]))))
            color = (*p["color"], alpha)
            text_surf = self.font_particle.render(p["text"], True, p["color"])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (p["x"], p["y"]))

    def _create_text_particle(self, text, pos, color):
        self.particles.append({
            "x": pos[0] + self.np_random.uniform(-10, 10),
            "y": pos[1] + self.np_random.uniform(-10, 10),
            "vy": self.np_random.uniform(0.5, 1.5),
            "text": text,
            "color": color,
            "life": 60,
            "lifespan": 60
        })

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _lerp_color(self, c1, c2, t):
        return (
            int(self._lerp(c1[0], c2[0], t)),
            int(self._lerp(c1[1], c2[1], t)),
            int(self._lerp(c1[2], c2[2], t)),
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Game loop
    while not terminated:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
        
        # --- Step Environment ---
        # For auto_advance=False, we only step on an action.
        # To make it playable, we can step on any key press or hold.
        if any(keys):
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
        else:
            # If no keys are pressed, we still need to render the current state
            obs = env._get_observation()

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()
    pygame.quit()