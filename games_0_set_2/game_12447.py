import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:54:41.651237
# Source Brief: brief_02447.md
# Brief Index: 2447
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import namedtuple

# Helper object for particles
Particle = namedtuple("Particle", ["pos", "vel", "radius", "color", "lifespan"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A fast-paced match-3 puzzle game. Swap adjacent tiles to create lines of three or more to score points before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to select a tile, then an adjacent one to swap. Press 'shift' to cancel a selection."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.TILE_TYPES = 6
        self.FPS = 30
        self.MAX_STEPS = 1800  # 60 seconds at 30 FPS
        self.WIN_SCORE = 1000

        # Visuals
        self.COLOR_BG = (20, 25, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.TILE_COLORS = [
            (220, 50, 50),    # Red
            (50, 220, 50),    # Green
            (80, 150, 255),   # Blue
            (240, 240, 80),   # Yellow
            (220, 80, 220),   # Magenta
            (255, 140, 50),   # Orange
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECT = (255, 255, 255, 100) # Transparent white
        self.COLOR_TEXT = (230, 230, 230)
        
        self.GRID_AREA_HEIGHT = 360
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_HEIGHT) / 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_HEIGHT) / 2
        self.TILE_SIZE = self.GRID_AREA_HEIGHT // self.GRID_SIZE
        self.TILE_PADDING = self.TILE_SIZE // 10

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_score = pygame.font.Font(None, 48)
        self.font_multiplier = pygame.font.Font(None, 40)
        self.font_gameover = pygame.font.Font(None, 80)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.bonus_multiplier = 1.0
        self.game_over = False
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.board_stable = True
        self.prev_action = [0, 0, 0]
        self.move_cooldown = 0
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.bonus_multiplier = 1.0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.animations = []
        self.particles = []
        self.board_stable = True
        self.prev_action = [0, 0, 0]
        self.move_cooldown = 0
        
        self._fill_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0

        self._update_bonus_multiplier()
        self._update_animations()
        self._update_particles()
        
        # Process input only when the board is not animating
        if not self.animations:
            if self.board_stable:
                reward += self._handle_input(action)
            else:
                # Board has settled after a fall, check for new matches (cascades)
                match_reward = self._find_and_process_matches()
                if match_reward > 0:
                    reward += match_reward
                    self._apply_gravity_and_refill()
                else:
                    self.board_stable = True # No more cascades

        # Store action for next step's press detection
        self.prev_action = action
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            reward += 100 if self.score >= self.WIN_SCORE else -100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Core Game Logic ---

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        space_pressed = space_val == 1 and self.prev_action[1] == 0
        shift_pressed = shift_val == 1 and self.prev_action[2] == 0
        reward = 0

        # --- Cursor Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement != 0 and self.move_cooldown == 0:
            # sfx: cursor_move.wav
            dx, dy = 0, 0
            if movement == 1: dy = -1
            elif movement == 2: dy = 1
            elif movement == 3: dx = -1
            elif movement == 4: dx = 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)
            self.move_cooldown = 4 # 4-frame cooldown

        # --- Cancel Selection ---
        if shift_pressed and self.selected_tile:
            # sfx: cancel.wav
            self.selected_tile = None
        
        # --- Select / Swap ---
        if space_pressed:
            cx, cy = self.cursor_pos
            if not self.selected_tile:
                # sfx: select.wav
                self.selected_tile = (cx, cy)
            else:
                sx, sy = self.selected_tile
                # Check for valid adjacent swap
                if abs(sx - cx) + abs(sy - cy) == 1:
                    # sfx: swap.wav
                    self._swap_tiles(sx, sy, cx, cy)
                    reward += 0.1 # Small reward for any valid action
                    self.board_stable = False
                else: # Invalid swap, treat as new selection
                    # sfx: select.wav
                    self.selected_tile = (cx, cy)
        return reward

    def _swap_tiles(self, x1, y1, x2, y2):
        # Swap in the data grid
        self.grid[y1][x1], self.grid[y2][y2] = self.grid[y2][y2], self.grid[y1][x1]
        
        # Check if the swap creates a match
        matches1 = self._get_matches_for_tile(x1, y1)
        matches2 = self._get_matches_for_tile(x2, y2)

        # If no match is created, swap back
        if not matches1 and not matches2:
            self.grid[y1][x1], self.grid[y2][y2] = self.grid[y2][y2], self.grid[y1][x1]
            # Create a "failed swap" animation (swap and swap back)
            self.animations.append({"type": "swap", "p1": (x1, y1), "p2": (x2, y2), "progress": 0, "duration": 8, "revert": True})
        else:
            # Create a successful swap animation
            self.animations.append({"type": "swap", "p1": (x1, y1), "p2": (x2, y2), "progress": 0, "duration": 8, "revert": False})
            self.selected_tile = None

    def _find_and_process_matches(self):
        all_matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                matches = self._get_matches_for_tile(c, r)
                if matches:
                    all_matches.update(matches)
        
        if all_matches:
            # sfx: match.wav
            num_matched = len(all_matches)
            reward = 5 * num_matched * self.bonus_multiplier
            self.score += int(10 * num_matched * self.bonus_multiplier)

            for r, c in all_matches:
                self._create_particles(c, r, self.grid[r][c])
                self.grid[r][c] = -1 # Mark for removal
            return reward
        return 0

    def _apply_gravity_and_refill(self):
        fall_animations = []
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != empty_row:
                        # Tile needs to fall
                        fall_animations.append({"type": "fall", "from": (c, r), "to": (c, empty_row), "progress": 0, "duration": 10})
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                    empty_row -= 1
            
            # Refill top with new tiles
            for r in range(empty_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(0, self.TILE_TYPES)
                fall_animations.append({"type": "fall", "from": (c, r - (empty_row + 1)), "to": (c, r), "progress": 0, "duration": 10})
        
        if fall_animations:
            # sfx: fall.wav
            self.animations.extend(fall_animations)
        self.board_stable = False


    # --- Update Methods ---

    def _update_animations(self):
        if not self.animations:
            if not self.board_stable:
                # Animations just finished, board might have new matches
                self.board_stable = False
            return

        finished_anims = []
        for anim in self.animations:
            anim["progress"] += 1
            if anim["progress"] >= anim["duration"]:
                if anim.get("revert", False) and anim["progress"] < anim["duration"] * 2:
                    # Continue animation for the revert trip
                    pass
                else:
                    finished_anims.append(anim)
        
        if finished_anims:
            self.animations = [a for a in self.animations if a not in finished_anims]
            if not self.animations:
                # All animations finished
                self.board_stable = False # Signal to check for matches next step

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for i, p in enumerate(self.particles):
            new_pos = (p.pos[0] + p.vel[0], p.pos[1] + p.vel[1])
            new_vel = (p.vel[0] * 0.95, p.vel[1] * 0.95 + 0.1) # Damping and gravity
            new_lifespan = p.lifespan - 1
            self.particles[i] = p._replace(pos=new_pos, vel=new_vel, lifespan=new_lifespan)

    def _update_bonus_multiplier(self):
        # Multiplier increases every 10 seconds (300 steps)
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.bonus_multiplier += 0.2
            # sfx: multiplier_up.wav

    # --- Rendering ---

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_HEIGHT, self.GRID_AREA_HEIGHT), border_radius=10)

        # Draw tiles
        self._render_tiles()

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.lifespan / 20.0))))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), int(p.radius), color)
            
        # Draw selection and cursor
        self._render_overlays()

    def _render_tiles(self):
        tile_render_size = self.TILE_SIZE - self.TILE_PADDING
        animating_tiles = set()
        
        # Process animations to find which tiles are moving
        for anim in self.animations:
            if anim["type"] == "swap":
                animating_tiles.add(anim["p1"])
                animating_tiles.add(anim["p2"])
                self._draw_animated_swap(anim, tile_render_size)
            elif anim["type"] == "fall":
                self._draw_animated_fall(anim, tile_render_size)
                animating_tiles.add(anim["to"])

        # Draw static tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (c, r) not in animating_tiles and self.grid[r][c] != -1:
                    tile_type = self.grid[r][c]
                    color = self.TILE_COLORS[tile_type]
                    screen_pos = self._get_tile_screen_pos(c, r)
                    rect = pygame.Rect(screen_pos[0], screen_pos[1], tile_render_size, tile_render_size)
                    pygame.draw.rect(self.screen, color, rect, border_radius=6)

    def _draw_animated_swap(self, anim, size):
        p1, p2 = anim["p1"], anim["p2"]
        progress = anim["progress"] / anim["duration"]
        
        if anim.get("revert", False):
            # Ping-pong interpolation for failed swap
            if progress > 1.0:
                progress = 2.0 - progress
        
        pos1_start, pos1_end = self._get_tile_screen_pos(*p1), self._get_tile_screen_pos(*p2)
        pos2_start, pos2_end = self._get_tile_screen_pos(*p2), self._get_tile_screen_pos(*p1)

        current_pos1 = (pos1_start[0] + (pos1_end[0] - pos1_start[0]) * progress, pos1_start[1] + (pos1_end[1] - pos1_start[1]) * progress)
        current_pos2 = (pos2_start[0] + (pos2_end[0] - pos2_start[0]) * progress, pos2_start[1] + (pos2_end[1] - pos2_start[1]) * progress)
        
        # After data swap, grid[p2] has tile from p1
        color1 = self.TILE_COLORS[self.grid[p2[1]][p2[0]]]
        color2 = self.TILE_COLORS[self.grid[p1[1]][p1[0]]]
        
        pygame.draw.rect(self.screen, color1, (*current_pos1, size, size), border_radius=6)
        pygame.draw.rect(self.screen, color2, (*current_pos2, size, size), border_radius=6)
        
    def _draw_animated_fall(self, anim, size):
        from_pos = self._get_tile_screen_pos(anim["from"][0], anim["from"][1])
        to_pos = self._get_tile_screen_pos(anim["to"][0], anim["to"][1])
        progress = anim["progress"] / anim["duration"]
        
        current_y = from_pos[1] + (to_pos[1] - from_pos[1]) * progress
        color = self.TILE_COLORS[self.grid[anim["to"][1]][anim["to"][0]]]
        
        pygame.draw.rect(self.screen, color, (to_pos[0], current_y, size, size), border_radius=6)

    def _render_overlays(self):
        # Draw selected tile highlight
        if self.selected_tile:
            sx, sy = self.selected_tile
            pos = self._get_tile_screen_pos(sx, sy, -self.TILE_PADDING / 2)
            size = self.TILE_SIZE
            
            # Pulsating effect
            pulse = abs(math.sin(self.steps * 0.2))
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            alpha = 70 + pulse * 50
            pygame.draw.rect(s, (*self.COLOR_SELECT[:3], alpha), (0, 0, size, size), border_radius=10)
            self.screen.blit(s, pos)

        # Draw cursor
        cx, cy = self.cursor_pos
        pos = self._get_tile_screen_pos(cx, cy, -self.TILE_PADDING // 2)
        size = self.TILE_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (*pos, size, size), width=3, border_radius=10)

    def _render_ui(self):
        # Score
        score_surf = self.font_score.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = (220, 50, 50) if time_left < 10 else self.COLOR_TEXT
        timer_surf = self.font_score.render(f"{time_left:.1f}", True, timer_color)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 20, 10))

        # Multiplier
        mult_str = f"x{self.bonus_multiplier:.1f}"
        mult_color_val = min(255, 100 + (self.bonus_multiplier - 1) * 100)
        mult_color = (255, 255, int(255-mult_color_val/2))
        mult_size = int(40 + (self.bonus_multiplier - 1) * 5)
        mult_font = pygame.font.Font(None, mult_size)
        mult_surf = mult_font.render(mult_str, True, mult_color)
        mult_rect = mult_surf.get_rect(center=(self.WIDTH / 2, 30))
        self.screen.blit(mult_surf, mult_rect)

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            text_surf = self.font_gameover.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    # --- Helper Methods ---

    def _fill_grid(self):
        self.grid = [[0] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                possible_types = list(range(self.TILE_TYPES))
                # Avoid creating matches on spawn
                if c >= 2 and self.grid[r][c-1] == self.grid[r][c-2]:
                    if self.grid[r][c-1] in possible_types:
                        possible_types.remove(self.grid[r][c-1])
                if r >= 2 and self.grid[r-1][c] == self.grid[r-2][c]:
                    if self.grid[r-1][c] in possible_types:
                        possible_types.remove(self.grid[r-1][c])
                
                self.grid[r][c] = self.np_random.choice(possible_types)

    def _get_matches_for_tile(self, c, r):
        if c < 0 or c >= self.GRID_SIZE or r < 0 or r >= self.GRID_SIZE:
            return None
        
        tile_type = self.grid[r][c]
        if tile_type == -1: return None

        # Horizontal
        h_matches = {(r, c)}
        # Left
        for i in range(c - 1, -1, -1):
            if self.grid[r][i] == tile_type: h_matches.add((r, i))
            else: break
        # Right
        for i in range(c + 1, self.GRID_SIZE):
            if self.grid[r][i] == tile_type: h_matches.add((r, i))
            else: break

        # Vertical
        v_matches = {(r, c)}
        # Up
        for i in range(r - 1, -1, -1):
            if self.grid[i][c] == tile_type: v_matches.add((i, c))
            else: break
        # Down
        for i in range(r + 1, self.GRID_SIZE):
            if self.grid[i][c] == tile_type: v_matches.add((i, c))
            else: break
        
        found_matches = set()
        if len(h_matches) >= 3: found_matches.update(h_matches)
        if len(v_matches) >= 3: found_matches.update(v_matches)
        
        return found_matches if found_matches else None
    
    def _get_tile_screen_pos(self, c, r, offset=0):
        x = self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_PADDING / 2 + offset
        y = self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_PADDING / 2 + offset
        return (x, y)

    def _create_particles(self, c, r, tile_type):
        center_x, center_y = self._get_tile_screen_pos(c, r)
        center_x += (self.TILE_SIZE - self.TILE_PADDING) / 2
        center_y += (self.TILE_SIZE - self.TILE_PADDING) / 2
        color = self.TILE_COLORS[tile_type]

        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(15, 25)
            self.particles.append(Particle((center_x, center_y), vel, radius, color, lifespan))

    def _check_termination(self):
        return self.timer <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

if __name__ == '__main__':
    # --- Manual Play ---
    # Re-enable display for manual testing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tile Matcher")
    
    running = True
    total_reward = 0
    
    # prev_action needs to be tracked for human play
    prev_action = [0, 0, 0]

    while running:
        # Action mapping for human play
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                prev_action = [0, 0, 0]

        # In manual play, we need to simulate the env's internal prev_action tracking
        # The env's step function will handle this for its next state, but for our key press detection,
        # we need to do it here.
        
        # A simple way to handle this is to pass the action and let the env do its thing.
        # The original code was missing tracking prev_action for the env.
        env.prev_action = prev_action
        obs, reward, terminated, truncated, info = env.step(action)
        prev_action = action # update for next frame
        total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to reset or quit
            reset_game = False
            while not reset_game and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        prev_action = [0, 0, 0]
                        reset_game = True
        
        env.clock.tick(env.FPS)
        
    pygame.quit()