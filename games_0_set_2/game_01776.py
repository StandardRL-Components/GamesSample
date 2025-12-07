
# Generated: 2025-08-27T18:14:51.751054
# Source Brief: brief_01776.md
# Brief Index: 1776

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move your character one tile at a time. Collect all the gold before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down stealth game. Sneak past patrolling guards in a grid-based world to collect 5 gold bars before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.TILE_SIZE = 40
        self.MAX_STEPS = 600
        self.NUM_GOLD = 5
        self.NUM_GUARDS = 3

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (50, 205, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)
        self.COLOR_GUARD = (255, 69, 0)
        self.COLOR_GUARD_GLOW = (255, 140, 100)
        self.COLOR_GUARD_VISION = (255, 69, 0, 60) # RGBA
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_GOLD_GLOW = (255, 255, 150)
        self.COLOR_UI_BG = (10, 15, 20, 200) # RGBA
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TIME_BAR = (40, 160, 220)
        self.COLOR_TIME_BAR_BG = (30, 80, 110)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.player_pos = (0, 0)
        self.guards = []
        self.gold_locations = []
        self.gold_collected = 0
        self.win_flash = 0
        self.lose_flash = 0
        self.last_reward_info = ""

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.gold_collected = 0
        self.win_flash = 0
        self.lose_flash = 0
        self.last_reward_info = ""

        # --- Procedural Level Generation ---
        all_tiles = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_tiles)

        # Player
        self.player_pos = all_tiles.pop()

        # Gold
        self.gold_locations = [all_tiles.pop() for _ in range(self.NUM_GOLD)]

        # Guards
        self.guards = []
        for _ in range(self.NUM_GUARDS):
            start_pos = all_tiles.pop()
            # Determine patrol axis (horizontal or vertical)
            if self.np_random.random() > 0.5: # Horizontal patrol
                path_len = self.np_random.integers(3, self.GRID_W // 2)
                end_x = min(self.GRID_W - 1, start_pos[0] + path_len)
                end_pos = (end_x, start_pos[1])
            else: # Vertical patrol
                path_len = self.np_random.integers(2, self.GRID_H // 2)
                end_y = min(self.GRID_H - 1, start_pos[1] + path_len)
                end_pos = (start_pos[0], end_y)
            
            # Ensure start and end are different if possible
            if start_pos == end_pos:
                if start_pos[0] > 0: end_pos = (start_pos[0]-1, start_pos[1])
                else: end_pos = (start_pos[0]+1, start_pos[1])

            self.guards.append({
                "pos": start_pos,
                "start": start_pos,
                "end": end_pos,
                "dir": 1 # 1 for forward, -1 for backward
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0.0
        terminated = False
        self.last_reward_info = ""

        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1
        
        # 1. Player Movement
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        
        px = np.clip(px, 0, self.GRID_W - 1)
        py = np.clip(py, 0, self.GRID_H - 1)
        self.player_pos = (px, py)
        
        # 2. Guard Movement
        for guard in self.guards:
            g_pos = list(guard["pos"])
            g_start = guard["start"]
            g_end = guard["end"]
            
            if g_start[0] != g_end[0]: # Horizontal
                g_pos[0] += guard["dir"]
                if g_pos[0] > max(g_start[0], g_end[0]) or g_pos[0] < min(g_start[0], g_end[0]):
                    guard["dir"] *= -1
                    g_pos[0] += 2 * guard["dir"] # Correct overshoot
            else: # Vertical
                g_pos[1] += guard["dir"]
                if g_pos[1] > max(g_start[1], g_end[1]) or g_pos[1] < min(g_start[1], g_end[1]):
                    guard["dir"] *= -1
                    g_pos[1] += 2 * guard["dir"] # Correct overshoot

            guard["pos"] = tuple(g_pos)

        # 3. Calculate Reward and Check State Changes
        reward += 0.1 # Survival reward
        
        # Check for gold collection
        if self.player_pos in self.gold_locations:
            self.gold_locations.remove(self.player_pos)
            self.gold_collected += 1
            reward += 10
            self.score += 10
            self.last_reward_info = "+10 GOLD!"
            # sfx: gold collect sound

        # Check for guard collision
        for guard in self.guards:
            if self.player_pos == guard["pos"]:
                reward = -100
                self.score -= 100
                terminated = True
                self.lose_flash = 15 # Flash red for 15 frames
                self.last_reward_info = "-100 CAUGHT!"
                # sfx: player caught sound
                break
        
        # 4. Check Termination Conditions
        if not terminated:
            if self.gold_collected == self.NUM_GOLD:
                reward += 100
                self.score += 100
                terminated = True
                self.win_flash = 15 # Flash green for 15 frames
                self.last_reward_info = "+100 VICTORY!"
                # sfx: victory fanfare
            elif self.time_remaining <= 0:
                reward = -50
                self.score -= 50
                terminated = True
                self.last_reward_info = "-50 TIMEOUT!"
                # sfx: timeout buzzer

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_gold()
        self._draw_guards()
        self._draw_player()
        self._draw_ui()
        
        if self.win_flash > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((self.COLOR_PLAYER_GLOW[0], self.COLOR_PLAYER_GLOW[1], self.COLOR_PLAYER_GLOW[2], 128))
            self.screen.blit(flash_surface, (0, 0))
            self.win_flash -= 1
        
        if self.lose_flash > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((self.COLOR_GUARD[0], self.COLOR_GUARD[1], self.COLOR_GUARD[2], 128))
            self.screen.blit(flash_surface, (0, 0))
            self.lose_flash -= 1

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold_collected": self.gold_collected,
            "time_remaining": self.time_remaining,
        }
        
    def _world_to_screen(self, x, y):
        return int(x * self.TILE_SIZE + self.TILE_SIZE / 2), int(y * self.TILE_SIZE + self.TILE_SIZE / 2)

    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
    def _draw_player(self):
        px, py = self._world_to_screen(*self.player_pos)
        radius = int(self.TILE_SIZE * 0.35)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER_GLOW)

    def _draw_guards(self):
        radius = int(self.TILE_SIZE * 0.35)
        for guard in self.guards:
            gx, gy = self._world_to_screen(*guard["pos"])
            # Body
            pygame.gfxdraw.filled_circle(self.screen, gx, gy, radius, self.COLOR_GUARD)
            pygame.gfxdraw.aacircle(self.screen, gx, gy, radius, self.COLOR_GUARD_GLOW)
            # Vision cone
            vision_length = self.TILE_SIZE * 1.5
            p1 = (gx, gy)
            if guard["start"][0] != guard["end"][0]: # Horizontal
                p2 = (gx + vision_length * guard["dir"], gy - self.TILE_SIZE / 2)
                p3 = (gx + vision_length * guard["dir"], gy + self.TILE_SIZE / 2)
            else: # Vertical
                p2 = (gx - self.TILE_SIZE / 2, gy + vision_length * guard["dir"])
                p3 = (gx + self.TILE_SIZE / 2, gy + vision_length * guard["dir"])
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_GUARD_VISION)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_GUARD_VISION)

    def _draw_gold(self):
        for gold_pos in self.gold_locations:
            gx, gy = self._world_to_screen(*gold_pos)
            w, h = int(self.TILE_SIZE * 0.6), int(self.TILE_SIZE * 0.3)
            rect = pygame.Rect(gx - w/2, gy - h/2, w, h)
            # Glow effect
            glow_rect = rect.inflate(6, 6)
            pygame.draw.rect(self.screen, self.COLOR_GOLD_GLOW, glow_rect, border_radius=5)
            # Main bar
            pygame.draw.rect(self.screen, self.COLOR_GOLD, rect, border_radius=3)
            
    def _draw_ui(self):
        ui_height = 40
        ui_surface = pygame.Surface((self.WIDTH, ui_height), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        
        # Score and Gold
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        gold_text = self.font_large.render(f"GOLD: {self.gold_collected}/{self.NUM_GOLD}", True, self.COLOR_GOLD)
        ui_surface.blit(score_text, (10, 8))
        ui_surface.blit(gold_text, (180, 8))

        # Reward Info
        if self.last_reward_info:
            reward_text = self.font_small.render(self.last_reward_info, True, self.COLOR_UI_TEXT)
            ui_surface.blit(reward_text, (self.WIDTH - 150, 12))

        # Time Bar
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.WIDTH // 2 - bar_w // 2, ui_height // 2 - bar_h // 2
        time_ratio = max(0, self.time_remaining / self.MAX_STEPS)
        pygame.draw.rect(ui_surface, self.COLOR_TIME_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        pygame.draw.rect(ui_surface, self.COLOR_TIME_BAR, (bar_x, bar_y, bar_w * time_ratio, bar_h), border_radius=5)

        self.screen.blit(ui_surface, (0, self.HEIGHT - ui_height))

    def close(self):
        pygame.quit()

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Stealth Grid")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # Start with no-op

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Get user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    print("Resetting environment...")
                    obs, info = env.reset()
                    action = np.array([0, 0, 0])
                    continue
                
                # Since it's turn-based, we step immediately on key press
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                action = np.array([0, 0, 0]) # Reset action after step

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit FPS for display
        
    print(f"Game Over! Final Info: {info}")
    env.close()