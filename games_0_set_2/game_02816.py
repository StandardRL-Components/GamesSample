import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select/deselect a crystal. "
        "Move a selected crystal into an empty adjacent space to cause matches."
    )

    game_description = (
        "A fast-paced isometric puzzle game. Shift glowing crystals to create matches of three or "
        "more against the clock. Clear the entire board to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 8, 8
        self.MAX_TIME = 60
        self.NUM_CRYSTAL_TYPES = 3

        # --- Visuals ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (40, 45, 60)
        self.CRYSTAL_BASE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
        ]
        self.CRYSTAL_SIDE_COLORS = [[c * 0.6 for c in color] for color in self.CRYSTAL_BASE_COLORS]
        self.CRYSTAL_TOP_COLORS = [[min(255, c * 1.4) for c in color] for color in self.CRYSTAL_BASE_COLORS]
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 220, 100)
        
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.TILE_THICKNESS = 18
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- State variables ---
        self.board = None
        self.cursor_pos = None
        self.selected_crystal = None
        self.time_left = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.particles = None
        self.prev_space_held = None
        self.background_polys = None # Will be generated on first reset
        
        # Note: self.reset() is not called here to avoid duplicate work if the user calls it immediately after __init__
        # The first call to reset() will fully initialize the environment.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.background_polys is None:
            self.background_polys = self._create_background_scenery()

        self.time_left = self.MAX_TIME
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.BOARD_WIDTH // 2, self.BOARD_HEIGHT // 2]
        self.selected_crystal = None
        self.particles = []
        self.prev_space_held = False
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Unpack action ---
        movement, space_held, shift_held = action
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        self.steps += 1
        self.time_left -= 1
        
        move_made = False
        
        # --- Handle Actions ---
        if space_press:
            if self.selected_crystal:
                self.selected_crystal = None
            elif self.board[self.cursor_pos[0], self.cursor_pos[1]] > 0:
                self.selected_crystal = list(self.cursor_pos)
        elif movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            
            if self.selected_crystal:
                sx, sy = self.selected_crystal
                tx, ty = sx + dx, sy + dy
                
                if 0 <= tx < self.BOARD_WIDTH and 0 <= ty < self.BOARD_HEIGHT and self.board[tx, ty] == 0:
                    self.board[tx, ty] = self.board[sx, sy]
                    self.board[sx, sy] = 0
                    self.selected_crystal = None
                    move_made = True
                else:
                    self.selected_crystal = None # Deselect on failed move
            else:
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.BOARD_WIDTH - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.BOARD_HEIGHT - 1)

        # --- Game Logic: Matches and Gravity ---
        if move_made:
            total_cleared = 0
            chain_bonus = 0
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                
                num_cleared = len(matches)
                total_cleared += num_cleared
                
                reward += num_cleared
                if num_cleared >= 5:
                    reward += 5 
                reward += chain_bonus
                
                for x, y in matches:
                    self._spawn_particles(x, y)
                    self.board[x, y] = 0
                
                self._apply_gravity()
                chain_bonus += 2 
            
            if total_cleared == 0:
                reward -= 0.2 
        elif movement != 0 or space_press:
            reward -= 0.1
        
        self.score += reward
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        crystals_left = np.count_nonzero(self.board)
        if crystals_left == 0:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.time_left <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "crystals_left": np.count_nonzero(self.board),
            "cursor_pos": list(self.cursor_pos),
            "selected": self.selected_crystal is not None,
        }

    # --- Helper and Rendering Methods ---

    def _generate_board(self):
        self.board = self.np_random.integers(
            1, self.NUM_CRYSTAL_TYPES + 1, size=(self.BOARD_WIDTH, self.BOARD_HEIGHT)
        )
        # Iteratively fix the board until no matches are found
        while matches := self._find_matches():
            for x, y in matches:
                # Replace the matched crystal with a new random one
                self.board[x, y] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _apply_gravity(self):
        for x in range(self.BOARD_WIDTH):
            empty_row = self.BOARD_HEIGHT - 1
            for y in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.board[x, y] != 0:
                    if y != empty_row:
                        self.board[x, empty_row] = self.board[x, y]
                        self.board[x, y] = 0
                    empty_row -= 1
                    
    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH - 2):
                color = self.board[x, y]
                if color > 0 and self.board[x + 1, y] == color and self.board[x + 2, y] == color:
                    run = set()
                    run.add((x, y))
                    run.add((x + 1, y))
                    run.add((x + 2, y))
                    # Check for longer runs
                    for i in range(x + 3, self.BOARD_WIDTH):
                        if self.board[i, y] == color:
                            run.add((i, y))
                        else:
                            break
                    matches.update(run)
        
        # Vertical matches
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT - 2):
                color = self.board[x, y]
                if color > 0 and self.board[x, y + 1] == color and self.board[x, y + 2] == color:
                    run = set()
                    run.add((x, y))
                    run.add((x, y + 1))
                    run.add((x, y + 2))
                    for i in range(y + 3, self.BOARD_HEIGHT):
                        if self.board[x, i] == color:
                            run.add((x, i))
                        else:
                            break
                    matches.update(run)
        return matches

    def _spawn_particles(self, x, y):
        cx, cy = self._iso_to_screen(x, y)
        cy += self.TILE_HEIGHT_HALF # Center of crystal
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 41)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': lifespan, 'max_life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_background_scenery(self):
        polys = []
        for i in range(15):
            color = (
                self.np_random.integers(20, 41),
                self.np_random.integers(25, 46),
                self.np_random.integers(40, 61),
            )
            points = []
            for _ in range(self.np_random.integers(3, 6)):
                points.append(
                    (
                        self.np_random.integers(-50, self.WIDTH + 51),
                        self.np_random.integers(-50, self.HEIGHT + 51),
                    )
                )
            polys.append({'color': color, 'points': points})
        return polys
    
    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        if self.background_polys:
            for poly in self.background_polys:
                pygame.gfxdraw.filled_polygon(self.screen, poly['points'], poly['color'])

        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                cx, cy = self._iso_to_screen(x, y)
                
                tile_points = [
                    (cx, cy),
                    (cx - self.TILE_WIDTH_HALF, cy + self.TILE_HEIGHT_HALF),
                    (cx, cy + self.TILE_HEIGHT_HALF * 2),
                    (cx + self.TILE_WIDTH_HALF, cy + self.TILE_HEIGHT_HALF)
                ]
                pygame.gfxdraw.aapolygon(self.screen, tile_points, self.COLOR_GRID)

                crystal_type = self.board[x, y]
                if crystal_type > 0:
                    color_idx = crystal_type - 1
                    base_color = self.CRYSTAL_BASE_COLORS[color_idx]
                    top_color = self.CRYSTAL_TOP_COLORS[color_idx]
                    side_color = self.CRYSTAL_SIDE_COLORS[color_idx]
                    
                    left_face = [tile_points[1], (tile_points[1][0], tile_points[1][1] + self.TILE_THICKNESS), (tile_points[2][0], tile_points[2][1] + self.TILE_THICKNESS), tile_points[2]]
                    right_face = [tile_points[3], (tile_points[3][0], tile_points[3][1] + self.TILE_THICKNESS), (tile_points[2][0], tile_points[2][1] + self.TILE_THICKNESS), tile_points[2]]
                    pygame.gfxdraw.filled_polygon(self.screen, left_face, side_color)
                    pygame.gfxdraw.filled_polygon(self.screen, right_face, side_color)
                    pygame.gfxdraw.aapolygon(self.screen, left_face, base_color)
                    pygame.gfxdraw.aapolygon(self.screen, right_face, base_color)

                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, top_color)
                    pygame.gfxdraw.aapolygon(self.screen, tile_points, base_color)

        cursor_cx, cursor_cy = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        cursor_points = [
            (cursor_cx, cursor_cy),
            (cursor_cx - self.TILE_WIDTH_HALF, cursor_cy + self.TILE_HEIGHT_HALF),
            (cursor_cx, cursor_cy + self.TILE_HEIGHT_HALF * 2),
            (cursor_cx + self.TILE_WIDTH_HALF, cursor_cy + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 3)

        if self.selected_crystal:
            sel_x, sel_y = self.selected_crystal
            sel_cx, sel_cy = self._iso_to_screen(sel_x, sel_y)
            
            pulse = (math.sin(self.steps * 0.4) + 1) / 2 
            radius = int(self.TILE_WIDTH_HALF * (1.2 + pulse * 0.4))
            alpha = int(100 + pulse * 50)
            glow_color = (*self.COLOR_CURSOR, alpha)

            glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (radius, radius), radius)
            self.screen.blit(glow_surf, (sel_cx - radius, sel_cy + self.TILE_HEIGHT_HALF - radius))

        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*self.COLOR_PARTICLE, int(alpha))
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                 # Note: pygame.draw.circle doesn't handle alpha well without a separate surface.
                 # This will draw solid circles that get smaller.
                 pygame.draw.circle(self.screen, self.COLOR_PARTICLE, p['pos'], size)

        crystals_left = np.count_nonzero(self.board)
        crystal_text = self.font_small.render(f"Crystals: {crystals_left}", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (10, 10))

        time_text = self.font_large.render(f"Time: {self.time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(midtop=(self.WIDTH // 2, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if crystals_left == 0:
                end_text = self.font_large.render("BOARD CLEARED!", True, (150, 255, 150))
            else:
                end_text = self.font_large.render("TIME'S UP!", True, (255, 150, 150))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: In headless mode ("dummy" video driver), no window will appear.
    # This is for testing the environment logic.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # To actually see the game, you would need to comment out the line
    # `os.environ["SDL_VIDEODRIVER"] = "dummy"` in __init__ and have a display.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Crystal Cavern")
        has_display = True
    except pygame.error:
        print("No display available. Running in headless mode.")
        has_display = False

    running = True
    terminated = False
    
    # Game loop
    while running:
        action_to_take = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                movement = 0
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                space_held = keys[pygame.K_SPACE]

                action_to_take = (movement, 1 if space_held else 0, 0)
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                    terminated = False
                    action_to_take = None # Don't step on reset
            if event.type == pygame.KEYUP:
                 if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                    action_to_take = (0, 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0, 0)


        if action_to_take and not terminated:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            print(f"Action: {action_to_take}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Crystals: {info['crystals_left']}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")

        if has_display:
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.clock.tick(30)

    pygame.quit()