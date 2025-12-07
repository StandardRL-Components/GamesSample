
# Generated: 2025-08-28T02:19:43.270694
# Source Brief: brief_04414.md
# Brief Index: 4414

        
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


class Tile:
    """Represents a single tile on the game board."""
    def __init__(self, value, pos, size):
        self.value = value
        self.rect = pygame.Rect(pos[0], pos[1], size, size)
        
        self.state = "hidden"  # hidden, revealing, revealed, hiding, matched
        self.anim_progress = 0.0
        self.anim_speed = 4.0 # determines speed of flip animation

    def update(self, dt):
        """Updates the tile's animation state."""
        if self.state in ["revealing", "hiding"]:
            self.anim_progress += self.anim_speed * dt
            if self.anim_progress >= 1.0:
                self.anim_progress = 1.0
                if self.state == "revealing":
                    self.state = "revealed"
                elif self.state == "hiding":
                    self.state = "hidden"
                self.anim_progress = 0.0

    def draw(self, surface, colors, symbols, font):
        """Draws the tile on the given surface."""
        # Animation scale factor for the flip effect
        scale = abs(math.cos(self.anim_progress * math.pi))
        
        # Determine which face to draw
        current_state = self.state
        if self.state == "revealing" and self.anim_progress > 0.5:
            current_state = "revealed"
        elif self.state == "hiding" and self.anim_progress <= 0.5:
            current_state = "revealed"

        if current_state == "hidden":
            color = colors["tile_hidden"]
            border_color = colors["tile_hidden_border"]
            temp_rect = self.rect.copy()
            temp_rect.width = int(self.rect.width * scale)
            temp_rect.centerx = self.rect.centerx
            pygame.draw.rect(surface, color, temp_rect, border_radius=8)
            pygame.draw.rect(surface, border_color, temp_rect, width=3, border_radius=8)
        
        elif current_state == "revealed":
            color = colors["tile_revealed_bg"]
            symbol_color = symbols[self.value]["color"]
            temp_rect = self.rect.copy()
            temp_rect.width = int(self.rect.width * scale)
            temp_rect.centerx = self.rect.centerx
            pygame.draw.rect(surface, color, temp_rect, border_radius=8)
            
            # Draw the symbol
            symbols[self.value]["draw_func"](surface, temp_rect, symbol_color)
            
            pygame.draw.rect(surface, symbol_color, temp_rect, width=4, border_radius=8)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to flip a tile."
    )

    game_description = (
        "A fast-paced grid-based memory game. Match all the pairs before time runs out!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.fps = 60 # Using 60fps for smoother visuals, time will be adjusted
        self.max_time_seconds = 60
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_BG = (44, 62, 80) # Dark blue-grey
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_CURSOR = (241, 196, 15) # Yellow
        self.COLOR_TEXT = (236, 240, 241) # White
        self.COLOR_TIMER_WARN = (231, 76, 60) # Red
        self.TILE_COLORS = {
            "tile_hidden": (127, 140, 141), # Grey
            "tile_hidden_border": (149, 165, 166),
            "tile_revealed_bg": (236, 240, 241), # Light grey
        }
        
        self._init_symbols()
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.board = []
        self.cursor_pos = [0, 0]
        self.cursor_vis_pos = [0, 0]
        self.revealed_tiles = []
        self.matched_values = []
        self.mismatch_timer = 0
        self.move_cooldown = 0
        self.prev_space_held = False
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def _init_symbols(self):
        """Initializes procedural drawing functions for tile symbols."""
        self.SYMBOLS = [
            {"color": (26, 188, 156), "draw_func": self._draw_circle},   # Turquoise
            {"color": (231, 76, 60), "draw_func": self._draw_square},    # Red
            {"color": (52, 152, 219), "draw_func": self._draw_triangle},  # Blue
            {"color": (155, 89, 182), "draw_func": self._draw_diamond},   # Purple
            {"color": (241, 196, 15), "draw_func": self._draw_cross},     # Yellow
            {"color": (46, 204, 113), "draw_func": self._draw_star},      # Green
            {"color": (230, 126, 34), "draw_func": self._draw_hexagon},   # Orange
            {"color": (255, 105, 180), "draw_func": self._draw_heart},    # Pink
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.max_time_seconds * self.fps
        
        self.cursor_pos = [0, 0]
        self.revealed_tiles = []
        self.matched_values = []
        self.mismatch_timer = 0
        self.move_cooldown = 0
        self.prev_space_held = False
        self.particles = []

        # Create board
        self.grid_size = (4, 4)
        self.board = [[None for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        tile_values = list(range(len(self.SYMBOLS))) * 2
        self.np_random.shuffle(tile_values)

        self.board_margin = 40
        self.tile_gap = 10
        self.grid_width = self.width - self.board_margin * 2
        self.grid_height = self.height - self.board_margin * 2
        self.tile_size = (self.grid_width - self.tile_gap * (self.grid_size[1] - 1)) // self.grid_size[1]
        
        start_x = (self.width - (self.tile_size * self.grid_size[1] + self.tile_gap * (self.grid_size[1] - 1))) // 2
        start_y = (self.height - (self.tile_size * self.grid_size[0] + self.tile_gap * (self.grid_size[0] - 1))) // 2
        self.board_offset = (start_x, start_y)
        
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                val = tile_values.pop()
                pos_x = start_x + c * (self.tile_size + self.tile_gap)
                pos_y = start_y + r * (self.tile_size + self.tile_gap)
                self.board[r][c] = Tile(val, (pos_x, pos_y), self.tile_size)
        
        # Initialize cursor visual position
        tile = self.board[self.cursor_pos[0]][self.cursor_pos[1]]
        self.cursor_vis_pos = list(tile.rect.center)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        dt = self.clock.tick(self.fps) / 1000.0
        reward = -0.01 # Small time penalty per step

        if not self.game_over:
            # Unpack action
            movement = action[0]
            space_held = action[1] == 1
            
            # --- Input Handling ---
            space_pressed = space_held and not self.prev_space_held
            self.prev_space_held = space_held
            
            reward += self._handle_input(movement, space_pressed)

            # --- Game State Update ---
            self.time_remaining -= 1
            if self.move_cooldown > 0: self.move_cooldown -= 1

            for r in range(self.grid_size[0]):
                for c in range(self.grid_size[1]):
                    self.board[r][c].update(dt)

            self._update_particles(dt)

            if self.mismatch_timer > 0:
                self.mismatch_timer -= 1
                if self.mismatch_timer == 0:
                    # Hide the mismatched tiles
                    for r_idx, c_idx in self.revealed_tiles:
                        self.board[r_idx][c_idx].state = "hiding"
                    self.revealed_tiles = []
            
            # Check for match
            if len(self.revealed_tiles) == 2 and self.mismatch_timer == 0:
                r1, c1 = self.revealed_tiles[0]
                r2, c2 = self.revealed_tiles[1]
                tile1 = self.board[r1][c1]
                tile2 = self.board[r2][c2]

                if tile1.value == tile2.value:
                    # MATCH
                    tile1.state = "matched"
                    tile2.state = "matched"
                    self.matched_values.append(tile1.value)
                    self.revealed_tiles = []
                    self.score += 25
                    reward += 10
                    # Sound: Positive match sound
                    self._create_particles(tile1.rect.center, self.SYMBOLS[tile1.value]["color"], 30)
                    self._create_particles(tile2.rect.center, self.SYMBOLS[tile2.value]["color"], 30)
                else:
                    # MISMATCH
                    self.mismatch_timer = int(0.75 * self.fps) # Wait 0.75s
                    self.score -= 5
                    reward -= 1
                    # Sound: Negative mismatch sound

        # --- Termination Check ---
        win_condition = len(self.matched_values) == len(self.SYMBOLS)
        loss_condition = self.time_remaining <= 0
        terminated = win_condition or loss_condition
        
        if terminated and not self.game_over:
            if win_condition:
                self.score += 100
                reward += 100
            else: # loss_condition
                self.score -= 50
                reward -= 50
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        """Processes player input and returns immediate rewards."""
        reward = 0
        # --- Cursor Movement ---
        if self.move_cooldown == 0 and movement != 0:
            r, c = self.cursor_pos
            if movement == 1: self.cursor_pos[0] = max(0, r - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.grid_size[0] - 1, r + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, c - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.grid_size[1] - 1, c + 1)
            if self.cursor_pos != [r, c]:
                self.move_cooldown = 5 # Cooldown in frames
                # Sound: Cursor move tick
        
        # --- Tile Selection ---
        if space_pressed:
            r, c = self.cursor_pos
            tile = self.board[r][c]
            if tile.state == "hidden" and len(self.revealed_tiles) < 2 and self.mismatch_timer == 0:
                tile.state = "revealing"
                self.revealed_tiles.append((r, c))
                reward += 0.1
                # Sound: Tile flip
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw tiles
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                tile = self.board[r][c]
                if tile.state != "matched":
                    tile.draw(self.screen, self.TILE_COLORS, self.SYMBOLS, self.font_ui)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))
            
        # Draw cursor
        target_tile = self.board[self.cursor_pos[0]][self.cursor_pos[1]]
        target_pos = list(target_tile.rect.center)
        
        # Smooth interpolation for cursor visual position
        self.cursor_vis_pos[0] += (target_pos[0] - self.cursor_vis_pos[0]) * 0.3
        self.cursor_vis_pos[1] += (target_pos[1] - self.cursor_vis_pos[1]) * 0.3

        cursor_rect = target_tile.rect.inflate(12, 12)
        cursor_rect.center = (int(self.cursor_vis_pos[0]), int(self.cursor_vis_pos[1]))
        
        # Pulsing glow effect for cursor
        glow_alpha = (math.sin(self.steps * 0.1) + 1) / 2 * 150 + 50
        s = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_CURSOR, glow_alpha), s.get_rect(), width=4, border_radius=12)
        self.screen.blit(s, cursor_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Timer
        time_sec = max(0, self.time_remaining // self.fps)
        time_milli = max(0, (self.time_remaining % self.fps) * (100 // self.fps))
        timer_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_ui.render(f"TIME: {time_sec:02d}:{time_milli:02d}", True, timer_color)
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 20, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = len(self.matched_values) == len(self.SYMBOLS)
            msg = "YOU WIN!" if win_condition else "TIME'S UP!"
            msg_render = self.font_game_over.render(msg, True, self.COLOR_CURSOR)
            msg_rect = msg_render.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(msg_render, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, self.time_remaining / self.fps),
            "pairs_matched": len(self.matched_values)
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 150)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.uniform(0.4, 0.8)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color, 'radius': self.np_random.uniform(2, 5)})

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            p['vel'][1] += 100 * dt # Gravity
            p['lifespan'] -= dt
            p['radius'] = max(0, p['radius'] * (p['lifespan'] / p['max_life']))
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    # --- Symbol Drawing Functions ---
    def _draw_circle(self, surface, rect, color):
        pygame.draw.circle(surface, color, rect.center, int(rect.width * 0.35))
    
    def _draw_square(self, surface, rect, color):
        s_rect = rect.inflate(-rect.width * 0.4, -rect.height * 0.4)
        pygame.draw.rect(surface, color, s_rect)

    def _draw_triangle(self, surface, rect, color):
        points = [
            (rect.centerx, rect.top + rect.height * 0.2),
            (rect.left + rect.width * 0.2, rect.bottom - rect.height * 0.2),
            (rect.right - rect.width * 0.2, rect.bottom - rect.height * 0.2),
        ]
        pygame.draw.polygon(surface, color, points)

    def _draw_diamond(self, surface, rect, color):
        points = [
            (rect.centerx, rect.top + rect.height * 0.15),
            (rect.left + rect.width * 0.15, rect.centery),
            (rect.centerx, rect.bottom - rect.height * 0.15),
            (rect.right - rect.width * 0.15, rect.centery),
        ]
        pygame.draw.polygon(surface, color, points)

    def _draw_cross(self, surface, rect, color):
        pygame.draw.line(surface, color, 
            (rect.left + rect.width * 0.25, rect.top + rect.height * 0.25),
            (rect.right - rect.width * 0.25, rect.bottom - rect.height * 0.25), 
            width=8)
        pygame.draw.line(surface, color, 
            (rect.right - rect.width * 0.25, rect.top + rect.height * 0.25),
            (rect.left + rect.width * 0.25, rect.bottom - rect.height * 0.25), 
            width=8)

    def _draw_star(self, surface, rect, color):
        n = 5
        center = rect.center
        radius1 = rect.width * 0.4
        radius2 = rect.width * 0.2
        points = []
        for i in range(n * 2):
            radius = radius1 if i % 2 == 0 else radius2
            angle = i * math.pi / n - math.pi / 2
            points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
        pygame.draw.polygon(surface, color, points)

    def _draw_hexagon(self, surface, rect, color):
        radius = rect.width * 0.38
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            points.append((rect.centerx + radius * math.cos(angle), rect.centery + radius * math.sin(angle)))
        pygame.draw.polygon(surface, color, points)

    def _draw_heart(self, surface, rect, color):
        # A bit more complex, using two circles and a triangle
        r = int(rect.width * 0.22)
        cx1 = rect.centerx - r
        cx2 = rect.centerx + r
        cy = rect.centery - r * 0.2
        pygame.draw.circle(surface, color, (cx1, cy), r)
        pygame.draw.circle(surface, color, (cx2, cy), r)
        bottom_point = (rect.centerx, rect.bottom - rect.height * 0.15)
        p1 = (cx1 - r, cy)
        p2 = (cx2 + r, cy)
        pygame.draw.polygon(surface, color, [p1, p2, bottom_point])

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Memory Grid")
    screen = pygame.display.set_mode((env.width, env.height))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input to Action Mapping ---
        movement = 0 # none
        space_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
    env.close()