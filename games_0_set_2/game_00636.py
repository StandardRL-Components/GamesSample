
# Generated: 2025-08-27T14:17:33.238268
# Source Brief: brief_00636.md
# Brief Index: 636

        
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
        "Controls: Arrow keys to move the cursor. Press Space or Shift to squash a bug."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced isometric arcade game. Squash all the bugs before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 50
    
    GRID_WIDTH = 15
    GRID_HEIGHT = 10
    TILE_WIDTH = 40
    TILE_HEIGHT = 20
    
    MAX_BUGS = 25
    MAX_TIME = 30  # seconds
    
    SPLAT_DURATION = 15 # frames

    # --- Colors ---
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    BUG_TYPES = {
        "green": {"color": (50, 220, 50), "points": 1},
        "blue": {"color": (50, 150, 255), "points": 2},
        "red": {"color": (255, 50, 50), "points": 3},
    }
    COLOR_SQUASHED = (80, 80, 80)


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
        
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Centering the grid
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT // 4)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.cursor_pos = [0, 0]
        self.bugs = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.bugs = []
        occupied_positions = set()
        bug_type_keys = list(self.BUG_TYPES.keys())

        for _ in range(self.MAX_BUGS):
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if pos not in occupied_positions:
                    occupied_positions.add(pos)
                    break
            
            # Weighted random choice for bug types (more common, less points)
            bug_type_name = self.np_random.choice(bug_type_keys, p=[0.5, 0.3, 0.2])
            bug_info = self.BUG_TYPES[bug_type_name]

            self.bugs.append({
                "grid_pos": list(pos),
                "type": bug_type_name,
                "color": bug_info["color"],
                "points": bug_info["points"],
                "squashed": False,
                "squash_timer": 0
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if self.game_over:
            return (
                self._get_observation(),
                0, # No reward after game over
                True,
                False,
                self._get_info()
            )

        self.steps += 1
        self.time_left = max(0, self.time_left - 1 / self.FPS)

        # --- Handle Actions ---
        movement = action[0]
        squash_action = action[1] == 1 or action[2] == 1
        
        # 1. Handle Movement and Movement Reward
        old_cursor_pos = self.cursor_pos[:]
        old_dist = self._get_dist_to_nearest_bug(old_cursor_pos)
        
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if tuple(old_cursor_pos) != tuple(self.cursor_pos):
            new_dist = self._get_dist_to_nearest_bug(self.cursor_pos)
            if new_dist is not None and old_dist is not None:
                if new_dist < old_dist:
                    reward += 0.1  # Moved closer
                else:
                    reward -= 0.01 # Moved further or same
        
        # 2. Handle Squash Action and Reward
        if squash_action:
            # sound: 'squash_attempt.wav'
            for bug in self.bugs:
                if not bug["squashed"] and bug["grid_pos"] == self.cursor_pos:
                    bug["squashed"] = True
                    bug["squash_timer"] = self.SPLAT_DURATION
                    self.score += bug["points"]
                    reward += bug["points"]
                    # sound: 'squash_success.wav'
                    break # Can only squash one bug per step

        # --- Check Termination Conditions ---
        squashed_count = sum(1 for b in self.bugs if b["squashed"])
        
        win = squashed_count == self.MAX_BUGS
        lose = self.time_left <= 0

        terminated = win or lose
        if terminated and not self.game_over:
            self.game_over = True
            if win:
                reward += 50 # Win bonus
                # sound: 'win_jingle.wav'
            else:
                reward -= 50 # Lose penalty
                # sound: 'lose_buzzer.wav'
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_bugs_and_splats()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "bugs_squashed": sum(1 for b in self.bugs if b["squashed"]),
            "bugs_total": self.MAX_BUGS,
        }

    # --- Helper and Rendering Methods ---

    def _grid_to_iso(self, gx, gy):
        iso_x = self.grid_origin_x + (gx - gy) * (self.TILE_WIDTH / 2)
        iso_y = self.grid_origin_y + (gx + gy) * (self.TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_iso(0, y)
            end = self._grid_to_iso(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_iso(x, 0)
            end = self._grid_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_bugs_and_splats(self):
        for bug in sorted(self.bugs, key=lambda b: b["grid_pos"][0] + b["grid_pos"][1]):
            sx, sy = self._grid_to_iso(bug["grid_pos"][0], bug["grid_pos"][1])
            sy += 5 # Center in tile
            
            if bug["squashed"]:
                if bug["squash_timer"] > 0:
                    progress = bug["squash_timer"] / self.SPLAT_DURATION
                    radius = int(self.TILE_WIDTH / 2.5 * progress)
                    # Use the original bug color for the splat, but fade it
                    splat_color = tuple(int(c * (0.5 + progress * 0.5)) for c in bug['color'])
                    pygame.gfxdraw.filled_ellipse(self.screen, sx, sy, radius, int(radius * 0.5), splat_color)
                    pygame.gfxdraw.aaellipse(self.screen, sx, sy, radius, int(radius * 0.5), splat_color)
                    bug["squash_timer"] -= 1
                else:
                    # Lingering grey splat
                    radius = int(self.TILE_WIDTH / 3)
                    pygame.gfxdraw.filled_ellipse(self.screen, sx, sy, radius, int(radius * 0.5), self.COLOR_SQUASHED)
            else:
                # Bobbing animation for live bugs
                bob = math.sin(self.steps * 0.2 + bug["grid_pos"][0]) * 3
                radius = int(self.TILE_WIDTH / 4)
                
                # Shadow
                shadow_pos = (sx, sy + 5)
                pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], radius, int(radius * 0.5), (0,0,0,100))
                
                # Bug body
                bug_pos = (sx, int(sy + bob))
                pygame.gfxdraw.filled_ellipse(self.screen, bug_pos[0], bug_pos[1], radius, radius, bug["color"])
                pygame.gfxdraw.aaellipse(self.screen, bug_pos[0], bug_pos[1], radius, radius, tuple(min(255, c+50) for c in bug["color"]))


    def _render_cursor(self):
        sx, sy = self_grid_to_iso = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        
        # Pulsating effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in self.COLOR_CURSOR)

        p1 = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        p2 = self._grid_to_iso(self.cursor_pos[0] + 1, self.cursor_pos[1])
        p3 = self._grid_to_iso(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1)
        p4 = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1] + 1)
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], color)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], color) # Draw twice for thickness

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Bugs squashed
        squashed_count = sum(1 for b in self.bugs if b["squashed"])
        bug_text = f"Bugs: {squashed_count}/{self.MAX_BUGS}"
        draw_text(bug_text, self.font_medium, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Timer
        timer_text = f"Time: {self.time_left:.1f}"
        timer_color = self.COLOR_TEXT if self.time_left > 5 else (255, 100, 100)
        text_surf = self.font_medium.render(timer_text, True, self.COLOR_TEXT)
        draw_text(timer_text, self.font_medium, timer_color, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10), self.COLOR_TEXT_SHADOW)

        # Score
        score_text = f"Score: {self.score}"
        text_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        draw_text(score_text, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT - 50), self.COLOR_TEXT_SHADOW)

        # Game Over Message
        if self.game_over:
            win = squashed_count == self.MAX_BUGS
            message = "YOU WIN!" if win else "TIME'S UP!"
            color = (100, 255, 100) if win else (255, 100, 100)
            text_surf = self.font_large.render(message, True, color)
            draw_text(message, self.font_large, color, (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50), self.COLOR_TEXT_SHADOW)


    def _get_dist_to_nearest_bug(self, from_pos):
        unsquashed_bugs = [b for b in self.bugs if not b["squashed"]]
        if not unsquashed_bugs:
            return None
        
        min_dist = float('inf')
        for bug in unsquashed_bugs:
            dist = math.hypot(from_pos[0] - bug["grid_pos"][0], from_pos[1] - bug["grid_pos"][1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Bug Squasher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("      Bug Squasher Controls")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping from keyboard ---
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
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print("Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Bugs Squashed: {info['bugs_squashed']}/{info['bugs_total']}")
    
    pygame.quit()