import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to flip a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced memory game. Find all the matching pairs of symbols before the time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    GRID_ROWS = 4
    GRID_COLS = 4
    GRID_MARGIN = 40
    GRID_LINE_WIDTH = 2
    TILE_SIZE = 80
    TILE_SPACING = 10
    TILE_BORDER_RADIUS = 8
    
    MAX_TIME = 60  # seconds
    MAX_STEPS = MAX_TIME * FPS

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_GRID_LINE = (50, 70, 90)
    COLOR_TILE_HIDDEN = (90, 110, 130)
    COLOR_TILE_REVEALED_BG = (40, 60, 80)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR = (0, 180, 220)
    COLOR_TIMER_BAR_WARN = (255, 100, 0)
    
    SHAPE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (80, 255, 255),  # Cyan
        (255, 80, 255),  # Magenta
        (255, 165, 0),   # Orange
        (128, 0, 128),   # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 24)
        self.font_large = pygame.font.SysFont("sans-serif", 48)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tiles = []
        self.mismatch_cooldown = 0
        self.prev_space_held = False
        self.particles = []
        self.matches_found = 0
        
        # Calculate grid position once
        grid_total_width = self.GRID_COLS * (self.TILE_SIZE + self.TILE_SPACING) - self.TILE_SPACING
        grid_total_height = self.GRID_ROWS * (self.TILE_SIZE + self.TILE_SPACING) - self.TILE_SPACING
        self.grid_top_left = (
            (self.SCREEN_WIDTH - grid_total_width) // 2,
            (self.SCREEN_HEIGHT - grid_total_height) // 2 + 20 # Offset for UI
        )
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_tiles = []
        self.mismatch_cooldown = 0
        self.prev_space_held = False
        self.particles = []
        self.matches_found = 0

        # Create and shuffle tiles
        num_pairs = (self.GRID_ROWS * self.GRID_COLS) // 2
        shape_ids = list(range(num_pairs)) * 2
        self.np_random.shuffle(shape_ids)
        
        self.grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                tile = {
                    "shape_id": shape_ids.pop(),
                    "state": "hidden",  # hidden, revealing, revealed, hiding, matched
                    "animation_progress": 0.0,
                    "pos": (r, c)
                }
                row.append(tile)
            self.grid.append(row)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = self.steps >= self.MAX_STEPS

        # --- Update Game Logic ---
        self.steps += 1
        self.timer = max(0, self.timer - 1 / self.FPS)
        
        # Update animations and cooldowns
        self._update_animations()
        if self.mismatch_cooldown > 0:
            self.mismatch_cooldown -= 1
            if self.mismatch_cooldown == 0:
                self._process_mismatch()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Handle player input
        moved = self._handle_movement(movement)
        if moved:
            reward -= 0.01 # Small penalty for movement

        is_reveal_action = space_held and not self.prev_space_held
        if is_reveal_action:
            revealed_tile = self._handle_reveal()
            if revealed_tile:
                reward += 0.1 # Small reward for revealing a new tile

        self.prev_space_held = space_held

        # Check for new match
        if len(self.selected_tiles) == 2:
            reward += self._check_for_match()
            self.selected_tiles = [] # Clear selection to prevent re-triggering

        # --- Check Termination ---
        if self.timer <= 0 and not self.game_over:
            self.game_over = True
            reward -= 10 # Penalty for timeout
        
        if self.matches_found == (self.GRID_ROWS * self.GRID_COLS) // 2 and not self.game_over:
            self.game_over = True
            reward += 100 # Big reward for winning
            self._create_particles((self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), (255,255,0), 200)

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if self.mismatch_cooldown > 0: return False # Lock controls during mismatch view
        
        moved = True
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS
        else:
            moved = False
        return moved

    def _handle_reveal(self):
        if len(self.selected_tiles) >= 2 or self.mismatch_cooldown > 0:
            return False

        r, c = self.cursor_pos
        tile = self.grid[r][c]

        if tile["state"] == "hidden":
            tile["state"] = "revealing"
            self.selected_tiles.append(tile)
            return True
        return False

    def _check_for_match(self):
        tile1, tile2 = self.selected_tiles
        
        if tile1["shape_id"] == tile2["shape_id"]:
            # --- MATCH ---
            tile1["state"] = "matched"
            tile2["state"] = "matched"
            self.matches_found += 1
            self.score += 10
            for tile in [tile1, tile2]:
                r, c = tile["pos"]
                px = self.grid_top_left[0] + c * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SIZE / 2
                py = self.grid_top_left[1] + r * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SIZE / 2
                self._create_particles((px, py), self.SHAPE_COLORS[tile["shape_id"]], 50)
            return 10.0
        else:
            # --- MISMATCH ---
            self.mismatch_cooldown = int(self.FPS * 0.75) # Wait 0.75s before flipping back
            for tile in [tile1, tile2]:
                r, c = tile["pos"]
                px = self.grid_top_left[0] + c * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SIZE / 2
                py = self.grid_top_left[1] + r * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SIZE / 2
                self._create_particles((px, py), (150, 0, 0), 20)
            return -1.0
    
    def _process_mismatch(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.grid[r][c]
                if tile["state"] == "revealed":
                    tile["state"] = "hiding"

    def _update_animations(self):
        # Update tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.grid[r][c]
                if tile["state"] == "revealing":
                    tile["animation_progress"] += 0.15
                    if tile["animation_progress"] >= 1.0:
                        tile["animation_progress"] = 1.0
                        tile["state"] = "revealed"
                elif tile["state"] == "hiding":
                    tile["animation_progress"] -= 0.15
                    if tile["animation_progress"] <= 0.0:
                        tile["animation_progress"] = 0.0
                        tile["state"] = "hidden"
        
        # Update particles
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 0.04
            p['vel'][1] += 0.1 # Gravity
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(
            self.grid_top_left[0] - self.TILE_SPACING,
            self.grid_top_left[1] - self.TILE_SPACING,
            self.GRID_COLS * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SPACING,
            self.GRID_ROWS * (self.TILE_SIZE + self.TILE_SPACING) + self.TILE_SPACING
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=self.TILE_BORDER_RADIUS)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self.grid[r][c]
                tile_rect = pygame.Rect(
                    self.grid_top_left[0] + c * (self.TILE_SIZE + self.TILE_SPACING),
                    self.grid_top_left[1] + r * (self.TILE_SIZE + self.TILE_SPACING),
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                # Flip animation
                progress = tile["animation_progress"]
                
                if tile["state"] != "matched":
                    # This logic handles the 3D flip effect
                    if progress == 0.0: # Fully hidden
                        pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, tile_rect, border_radius=self.TILE_BORDER_RADIUS)
                    elif progress == 1.0: # Fully revealed
                        pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED_BG, tile_rect, border_radius=self.TILE_BORDER_RADIUS)
                        self._draw_shape(self.screen, tile["shape_id"], tile_rect)
                    else: # Mid-animation
                        anim_rect = tile_rect.copy()
                        scale = abs(math.cos(progress * math.pi))
                        anim_rect.width = int(self.TILE_SIZE * scale)
                        anim_rect.centerx = tile_rect.centerx
                        
                        if progress < 0.5: # Showing back
                             pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, anim_rect, border_radius=self.TILE_BORDER_RADIUS)
                        else: # Showing front
                             pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED_BG, anim_rect, border_radius=self.TILE_BORDER_RADIUS)
                             self._draw_shape(self.screen, tile["shape_id"], anim_rect)

                else: # Matched tile is faded
                    pygame.draw.rect(self.screen, self.COLOR_GRID_BG, tile_rect, border_radius=self.TILE_BORDER_RADIUS)

        # Draw cursor
        cur_r, cur_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_top_left[0] + cur_c * (self.TILE_SIZE + self.TILE_SPACING) - 4,
            self.grid_top_left[1] + cur_r * (self.TILE_SIZE + self.TILE_SPACING) - 4,
            self.TILE_SIZE + 8, self.TILE_SIZE + 8
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=self.TILE_BORDER_RADIUS+4)

        # Draw particles
        self._update_and_draw_particles()

    def _draw_shape(self, surface, shape_id, rect):
        if rect.width <= 0: return # Don't draw on a zero-width surface
        color = self.SHAPE_COLORS[shape_id % len(self.SHAPE_COLORS)]
        center = rect.center
        size = int(rect.width * 0.35)
        if size <= 0: return

        if shape_id == 0: # Circle
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], size, color)
            pygame.gfxdraw.aacircle(surface, center[0], center[1], size, color)
        elif shape_id == 1: # Square
            shape_rect = pygame.Rect(0, 0, size*2, size*2)
            shape_rect.center = center
            pygame.draw.rect(surface, color, shape_rect, border_radius=4)
        elif shape_id == 2: # Triangle
            points = [
                (center[0], center[1] - size),
                (center[0] - size, center[1] + size),
                (center[0] + size, center[1] + size)
            ]
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)
        elif shape_id == 3: # Cross
            pygame.draw.line(surface, color, (center[0]-size, center[1]-size), (center[0]+size, center[1]+size), 10)
            pygame.draw.line(surface, color, (center[0]-size, center[1]+size), (center[0]+size, center[1]-size), 10)
        elif shape_id == 4: # Diamond
            points = [
                (center[0], center[1] - int(size*1.2)),
                (center[0] + int(size*1.2), center[1]),
                (center[0], center[1] + int(size*1.2)),
                (center[0] - int(size*1.2), center[1])
            ]
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)
        elif shape_id == 5: # Star
            n = 5
            points = []
            for i in range(2 * n):
                radius = size * 1.5 if i % 2 == 0 else size * 0.7
                angle = i * math.pi / n - math.pi / 2
                points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)
        elif shape_id == 6: # Hexagon
            points = []
            for i in range(6):
                angle = i * math.pi / 3
                points.append((center[0] + size * math.cos(angle), center[1] + size * math.sin(angle)))
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)
        elif shape_id == 7: # Donut
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], size, color)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(size*0.5), self.COLOR_TILE_REVEALED_BG)

    def _render_ui(self):
        # Score / Matches
        matches_text = f"MATCHES: {self.matches_found} / {self.GRID_ROWS * self.GRID_COLS // 2}"
        text_surf = self.font_main.render(matches_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 10))

        # Timer Bar
        bar_width = 200
        bar_height = 20
        time_ratio = self.timer / self.MAX_TIME
        current_bar_width = int(bar_width * time_ratio)
        bar_color = self.COLOR_TIMER_BAR if time_ratio > 0.25 else self.COLOR_TIMER_BAR_WARN
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.SCREEN_WIDTH - bar_width - 20, 15, bar_width, bar_height), border_radius=5)
        if current_bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (self.SCREEN_WIDTH - bar_width - 20, 15, current_bar_width, bar_height), border_radius=5)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': 1.0,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            alpha = int(255 * p['life'])
            color = (*p['color'], alpha)
            # A bit of a hack to draw transparent rects
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "matches_found": self.matches_found
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test reset. This must be done first to initialize the game state.
        obs, info = self.reset()
        
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (using the observation from reset)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
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
    # To play, you need a display. Comment out the os.environ line at the top.
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Memory Grid")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Need to track spacebar press state for manual play
    space_was_pressed = False

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # The environment expects a "held" state, not a "pressed" event
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            # Optional: add a delay before restarting
            pygame.time.wait(2000)
            obs, info = env.reset()


        clock.tick(GameEnv.FPS)

    pygame.quit()