
# Generated: 2025-08-27T18:00:58.375398
# Source Brief: brief_01706.md
# Brief Index: 1706

        
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
        "Controls: Arrow keys to move cursor. Space to plant on empty soil, harvest mature crops, or sell at the barn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Maximize profits by rapidly planting, harvesting, and selling crops in a fast-paced top-down farming simulator."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    # Timing
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    CROP_GROW_TIME_SECONDS = 2
    CROP_GROW_STEPS = CROP_GROW_TIME_SECONDS * FPS

    # Grid
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    
    # Rewards
    REWARD_WIN = 100
    REWARD_LOSE = -10
    REWARD_HARVEST = 0.1
    REWARD_SELL = 1.0
    REWARD_ACTION_FAIL = -0.01

    # Goal
    WIN_MONEY = 1000
    CROP_SELL_PRICE = 10

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_SOIL = (101, 67, 33)
    COLOR_PLANTED_SEED = (152, 251, 152) # PaleGreen
    COLOR_GROWN_CROP = (255, 215, 0) # Gold
    COLOR_CURSOR = (255, 255, 255, 100) # Semi-transparent white
    COLOR_BARN = (178, 34, 34) # Firebrick
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_SHADOW = (10, 10, 10)
    
    # Cell States
    STATE_EMPTY = 0
    STATE_PLANTED = 1
    STATE_GROWN = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Screen layout
        self.GRID_AREA_RECT = pygame.Rect(40, 40, 480, 320)
        self.CELL_WIDTH = self.GRID_AREA_RECT.width // self.GRID_WIDTH
        self.CELL_HEIGHT = self.GRID_AREA_RECT.height // self.GRID_HEIGHT
        self.BARN_RECT = pygame.Rect(self.GRID_AREA_RECT.right + 20, self.GRID_AREA_RECT.top, 80, self.GRID_AREA_RECT.height)
        
        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.money = 0
        self.inventory = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.win_message = ""
        self.prev_space_held = False
        self.particles = []
        self.rng = np.random.default_rng()

        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid = [[{'state': self.STATE_EMPTY, 'timer': 0} for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.money = 0
        self.inventory = 0
        self.steps = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.win_message = ""
        self.prev_space_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            self.time_remaining = self.GAME_DURATION_SECONDS - (self.steps / self.FPS)

            # --- Update Game Logic ---
            self._update_crops()
            
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            self._handle_movement(movement)
            reward += self._handle_action(space_held)
            
            self.prev_space_held = space_held
            self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.money >= self.WIN_MONEY:
                reward += self.REWARD_WIN
                self.win_message = "YOU WIN!"
            else:
                reward += self.REWARD_LOSE
                self.win_message = "TIME'S UP!"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_crops(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell = self.grid[y][x]
                if cell['state'] == self.STATE_PLANTED:
                    cell['timer'] -= 1
                    if cell['timer'] <= 0:
                        cell['state'] = self.STATE_GROWN
                        # sfx: crop grown chime
                        self._spawn_particles(x, y, self.COLOR_GROWN_CROP, 5, 1.5)

    def _handle_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        # Clamp cursor position to grid + barn area
        self.cursor_pos[0] = max(0, min(self.GRID_WIDTH, self.cursor_pos[0]))
        self.cursor_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1]))

    def _handle_action(self, space_held):
        action_reward = 0
        space_pressed = space_held and not self.prev_space_held
        
        if not space_pressed:
            return 0

        cx, cy = self.cursor_pos

        # Check for Barn interaction
        if cx == self.GRID_WIDTH:
            if self.inventory > 0:
                # sfx: cash register
                self.money += self.inventory * self.CROP_SELL_PRICE
                action_reward += self.REWARD_SELL * self.inventory
                self.inventory = 0
            else:
                # sfx: action failed buzz
                action_reward += self.REWARD_ACTION_FAIL
            return action_reward

        # Grid interactions
        cell = self.grid[cy][cx]
        if cell['state'] == self.STATE_EMPTY:
            # Plant a seed
            # sfx: planting sound
            cell['state'] = self.STATE_PLANTED
            cell['timer'] = self.CROP_GROW_STEPS
            self._spawn_particles(cx, cy, self.COLOR_SOIL, 10, 1)
        elif cell['state'] == self.STATE_GROWN:
            # Harvest a crop
            # sfx: harvesting pop
            cell['state'] = self.STATE_EMPTY
            self.inventory += 1
            action_reward += self.REWARD_HARVEST
            self._spawn_particles(cx, cy, self.COLOR_GROWN_CROP, 20, 2)
        else: # Tried to act on a planted but not grown cell
            # sfx: action failed buzz
            action_reward += self.REWARD_ACTION_FAIL
            
        return action_reward

    def _check_termination(self):
        return self.time_remaining <= 0 or self.money >= self.WIN_MONEY

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.money,
            "steps": self.steps,
            "inventory": self.inventory,
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_rect = pygame.Rect(
                    self.GRID_AREA_RECT.left + x * self.CELL_WIDTH,
                    self.GRID_AREA_RECT.top + y * self.CELL_HEIGHT,
                    self.CELL_WIDTH, self.CELL_HEIGHT
                )
                pygame.draw.rect(self.screen, self.COLOR_SOIL, cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, cell_rect, 1)
                
                cell = self.grid[y][x]
                center_x, center_y = cell_rect.center
                
                if cell['state'] == self.STATE_PLANTED:
                    progress = 1.0 - (cell['timer'] / self.CROP_GROW_STEPS)
                    radius = int(progress * (self.CELL_WIDTH / 2 - 4))
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLANTED_SEED)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLANTED_SEED)
                elif cell['state'] == self.STATE_GROWN:
                    radius = int(self.CELL_WIDTH / 2 - 4)
                    # Pulsing glow effect
                    glow_radius = radius + int(3 * (1 + math.sin(self.steps * 0.2)))
                    glow_color = (*self.COLOR_GROWN_CROP, 50)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, glow_color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, glow_color)
                    
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_GROWN_CROP)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GROWN_CROP)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

        # Draw Barn
        pygame.draw.rect(self.screen, self.COLOR_BARN, self.BARN_RECT)
        self._render_text("BARN", self.font_medium, self.BARN_RECT.center, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, align="center")

        # Draw Cursor
        cx, cy = self.cursor_pos
        if cx < self.GRID_WIDTH:
            cursor_rect = pygame.Rect(
                self.GRID_AREA_RECT.left + cx * self.CELL_WIDTH,
                self.GRID_AREA_RECT.top + cy * self.CELL_HEIGHT,
                self.CELL_WIDTH, self.CELL_HEIGHT
            )
        else: # Cursor is over the barn
            cursor_rect = self.BARN_RECT
        
        s = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, cursor_rect, 2)
    
    def _render_ui(self):
        # Timer
        time_str = f"{max(0, self.time_remaining):.1f}s"
        self._render_text(time_str, self.font_medium, (20, 10), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        # Money
        money_str = f"${self.money}"
        self._render_text(money_str, self.font_medium, (620, 10), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, align="topright")

        # Inventory
        inv_str = f"Crops: {self.inventory}"
        self._render_text(inv_str, self.font_small, (620, 45), self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW, align="topright")

    def _render_game_over(self):
        overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        self._render_text(self.win_message, self.font_large, (320, 200), self.COLOR_GROWN_CROP, self.COLOR_UI_SHADOW, align="center")

    def _render_text(self, text, font, pos, color, shadow_color, align="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topright":
            text_rect.topright = pos
        else: # topleft
            text_rect.topleft = pos

        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)
        
    def _spawn_particles(self, grid_x, grid_y, color, count, max_speed):
        cell_rect = pygame.Rect(
            self.GRID_AREA_RECT.left + grid_x * self.CELL_WIDTH,
            self.GRID_AREA_RECT.top + grid_y * self.CELL_HEIGHT,
            self.CELL_WIDTH, self.CELL_HEIGHT
        )
        center_x, center_y = cell_rect.center

        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * max_speed
            self.particles.append({
                'x': center_x,
                'y': center_y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'lifetime': self.rng.integers(15, 30),
                'color': color,
                'size': self.rng.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['size'] > 0]

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Arcade Farmer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Score: ${info['score']}")
            # The environment handles the game over screen, so we just wait for reset
            
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    pygame.quit()