
# Generated: 2025-08-28T05:57:11.163896
# Source Brief: brief_05740.md
# Brief Index: 5740

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to pick up or drop a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a crystal maze. Place crystals to illuminate all paths and escape the caverns before time runs out."
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
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals & Game Constants
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH_UNLIT = (70, 80, 100)
        self.COLOR_PATH_LIT = (255, 220, 100)
        self.COLOR_PATH_LIT_GLOW = (200, 150, 50, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_GLOW = (255, 255, 255, 100)
        self.CRYSTAL_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (100, 255, 100), # Light Green
        ]
        
        self.GRID_SIZE = (12, 12)
        self.TILE_WIDTH = 40
        self.TILE_HEIGHT = self.TILE_WIDTH // 2
        self.ORIGIN_X = self.screen_width // 2
        self.ORIGIN_Y = 100
        
        self.MAX_STEPS = 600

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.crystals = None
        self.held_crystal_idx = None
        self.paths = None
        self.total_segments = 0
        
        # Initialize state variables before validation
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def _world_to_iso(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        iso_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        iso_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(iso_x), int(iso_y)

    def _generate_layout(self):
        """Generates a random but solvable puzzle layout."""
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.held_crystal_idx = None

        # Generate paths
        self.paths = []
        self.total_segments = 0
        path_templates = [
            [(1, 1), (1, 6), (4, 6), (4, 9), (2, 9)],
            [(10, 2), (7, 2), (7, 5), (9, 5)],
            [(2, 10), (2, 8), (5, 8), (5, 10), (8, 10), (8, 7), (10, 7), (10, 10)],
        ]
        
        for template in path_templates:
            path_obj = {"segments": [], "is_lit": []}
            for i in range(len(template) - 1):
                p1 = template[i]
                p2 = template[i+1]
                
                # Create segments between points
                cx, cy = p1
                nx, ny = p2
                while (cx, cy) != (nx, ny):
                    next_cx, next_cy = cx, cy
                    if cx < nx: next_cx += 1
                    elif cx > nx: next_cx -= 1
                    if cy < ny: next_cy += 1
                    elif cy > ny: next_cy -= 1
                    path_obj["segments"].append(((cx, cy), (next_cx, next_cy)))
                    path_obj["is_lit"].append(False)
                    cx, cy = next_cx, next_cy
            
            self.paths.append(path_obj)
            self.total_segments += len(path_obj["segments"])

        # Generate crystals
        self.crystals = []
        occupied_pos = set()
        for path_group in self.paths:
            for seg in path_group["segments"]:
                occupied_pos.add(seg[0])
                occupied_pos.add(seg[1])
        
        for i in range(len(self.CRYSTAL_COLORS)):
            while True:
                pos = (
                    self.np_random.integers(1, self.GRID_SIZE[0] - 1),
                    self.np_random.integers(1, self.GRID_SIZE[1] - 1)
                )
                if pos not in occupied_pos:
                    self.crystals.append({
                        "pos": list(pos),
                        "color": self.CRYSTAL_COLORS[i],
                    })
                    occupied_pos.add(pos)
                    break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_layout()
        self._update_path_illumination()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean, only care about press, not hold
        shift_held = action[2] == 1  # Boolean, unused in this design
        
        self.steps += 1
        reward = -0.02  # Time penalty

        # --- Game Logic ---
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE[1] - 1)
        
        # 2. Handle crystal interaction
        crystal_moved = False
        if space_pressed:
            if self.held_crystal_idx is not None:
                # Try to drop the crystal
                can_drop = True
                # Check if dropping on another crystal
                for i, c in enumerate(self.crystals):
                    if i != self.held_crystal_idx and c["pos"] == self.cursor_pos:
                        can_drop = False
                        break
                if can_drop:
                    # sfx: crystal_drop
                    self.crystals[self.held_crystal_idx]["pos"] = list(self.cursor_pos)
                    self.held_crystal_idx = None
                    crystal_moved = True
            else:
                # Try to pick up a crystal
                for i, c in enumerate(self.crystals):
                    if c["pos"] == self.cursor_pos:
                        # sfx: crystal_pickup
                        self.held_crystal_idx = i
                        break
        
        # 3. Update illumination and calculate rewards if a crystal was moved
        if crystal_moved:
            prev_lit_count = self._get_lit_segment_count()
            prev_completed_paths = self._get_completed_path_count()

            self._update_path_illumination()
            
            new_lit_count = self._get_lit_segment_count()
            new_completed_paths = self._get_completed_path_count()
            
            # Reward for newly lit segments
            reward += (new_lit_count - prev_lit_count) * 1.0
            if new_lit_count > prev_lit_count:
                pass # sfx: path_lit_segment
            
            # Reward for completing a whole path
            reward += (new_completed_paths - prev_completed_paths) * 5.0
            if new_completed_paths > prev_completed_paths:
                pass # sfx: path_lit_full

        # 4. Check for termination
        terminated = False
        win = self._get_lit_segment_count() == self.total_segments
        timeout = self.steps >= self.MAX_STEPS

        if win:
            # sfx: win_game
            reward += 100
            terminated = True
            self.game_over = True
        elif timeout:
            # sfx: lose_game
            reward += -100
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_path_illumination(self):
        """Calculates which path segments are lit by crystals."""
        for path_group in self.paths:
            for i in range(len(path_group["segments"])):
                path_group["is_lit"][i] = False

        for crystal in self.crystals:
            cx, cy = crystal["pos"]
            for path_group in self.paths:
                for i, seg in enumerate(path_group["segments"]):
                    (p1x, p1y), (p2x, p2y) = seg
                    # Check for vertical segment on the crystal's column
                    if p1x == p2x and p1x == cx:
                        path_group["is_lit"][i] = True
                    # Check for horizontal segment on the crystal's row
                    if p1y == p2y and p1y == cy:
                        path_group["is_lit"][i] = True

    def _get_lit_segment_count(self):
        return sum(s for pg in self.paths for s in pg["is_lit"])

    def _get_completed_path_count(self):
        count = 0
        for pg in self.paths:
            if all(pg["is_lit"]):
                count += 1
        return count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        """Renders all game world elements."""
        
        # Render paths (draw lit paths last to be on top)
        for lit_pass in [False, True]:
            for path_group in self.paths:
                for i, seg in enumerate(path_group["segments"]):
                    is_lit = path_group["is_lit"][i]
                    if is_lit != lit_pass:
                        continue
                    
                    p1 = self._world_to_iso(seg[0][0], seg[0][1])
                    p2 = self._world_to_iso(seg[1][0], seg[1][1])
                    color = self.COLOR_PATH_LIT if is_lit else self.COLOR_PATH_UNLIT
                    
                    if is_lit:
                        # Glow effect
                        pygame.draw.line(self.screen, self.COLOR_PATH_LIT_GLOW, p1, p2, 7)
                    pygame.draw.line(self.screen, color, p1, p2, 3)

        # Render crystals
        for i, crystal in enumerate(self.crystals):
            if self.held_crystal_idx == i:
                continue # Don't draw held crystal on the grid
            
            cx, cy = self._world_to_iso(crystal["pos"][0], crystal["pos"][1])
            points = [
                (cx, cy - self.TILE_HEIGHT / 2),
                (cx + self.TILE_WIDTH / 2, cy),
                (cx, cy + self.TILE_HEIGHT / 2),
                (cx - self.TILE_WIDTH / 2, cy),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, crystal["color"])
            pygame.gfxdraw.aapolygon(self.screen, points, crystal["color"])

        # Render cursor and held crystal
        cursor_iso_x, cursor_iso_y = self._world_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, cursor_iso_x, cursor_iso_y, 10, self.COLOR_CURSOR_GLOW)
        # Cursor
        pygame.gfxdraw.aacircle(self.screen, cursor_iso_x, cursor_iso_y, 5, self.COLOR_CURSOR)
        pygame.gfxdraw.filled_circle(self.screen, cursor_iso_x, cursor_iso_y, 5, self.COLOR_CURSOR)

        if self.held_crystal_idx is not None:
            crystal = self.crystals[self.held_crystal_idx]
            held_y_offset = -25
            cx, cy = cursor_iso_x, cursor_iso_y + held_y_offset
            points = [
                (cx, cy - self.TILE_HEIGHT / 2),
                (cx + self.TILE_WIDTH / 2, cy),
                (cx, cy + self.TILE_HEIGHT / 2),
                (cx - self.TILE_WIDTH / 2, cy),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, crystal["color"])
            pygame.gfxdraw.aapolygon(self.screen, points, crystal["color"])

    def _render_ui(self):
        """Renders the UI overlay."""
        # Paths Lit
        lit_count = self._get_lit_segment_count()
        path_text = f"Paths Lit: {lit_count} / {self.total_segments}"
        text_surf = self.font_large.render(path_text, True, self.COLOR_PATH_LIT)
        self.screen.blit(text_surf, (20, 20))

        # Time remaining
        time_left = self.MAX_STEPS - self.steps
        time_text = f"Time: {time_left}"
        text_surf = self.font_large.render(time_text, True, self.COLOR_CURSOR)
        text_rect = text_surf.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(text_surf, text_rect)
        
        if self.game_over:
            win = self._get_lit_segment_count() == self.total_segments
            message = "All Paths Illuminated!" if win else "Time Ran Out"
            color = self.COLOR_PATH_LIT if win else (255, 50, 50)
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "lit_segments": self._get_lit_segment_count(),
            "total_segments": self.total_segments,
        }

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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Set up the display window
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Crystal Caverns")

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # Event handling for manual play
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        
        if terminated:
            # In a terminated state, wait for reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and (event.key == pygame.K_r or event.key == pygame.K_SPACE):
                    obs, info = env.reset()
                    terminated = False
            
            # Keep rendering the final state
            frame = env._get_observation()
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(10) # Lower FPS when game is over
            continue

        # Map keyboard inputs to the action space for manual play
        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # We only care about the "press" event for space, not hold.
        # The main loop is fast, so this check is sufficient for manual play.
        if keys[pygame.K_SPACE]: action[1] = 1
        
        # Since auto_advance is False, we only step when there's an action.
        # For a smoother manual play experience, we'll step on every frame.
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(10) # Control the speed of manual play (10 steps per second)

    env.close()