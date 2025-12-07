
# Generated: 2025-08-28T01:31:07.031638
# Source Brief: brief_04126.md
# Brief Index: 4126

        
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

    user_guide = "Controls: Use arrow keys to move the cursor. Press Space to select a gem, then move to an adjacent gem and press Space again to swap."
    game_description = "Swap adjacent gems in an 8x8 grid to create matches of 3 or more. Reach 100 gems collected within 20 moves to win!"

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.GRID_WIDTH = 8
        self.GRID_HEIGHT = 8
        self.NUM_GEM_TYPES = 6
        self.CELL_SIZE = 48
        self.GRID_LINE_WIDTH = 2
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) + 10

        self.MAX_MOVES = 20
        self.GEM_GOAL = 100
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (40, 50, 60)
        self.COLOR_GRID_LINES = (60, 70, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255, 128)
        self.COLOR_SELECTED = (255, 255, 0, 192)
        
        self.GEM_COLORS = [
            (255, 50, 50),   # Red
            (50, 255, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 255, 50),  # Yellow
            (200, 50, 255),  # Purple
            (255, 150, 50),  # Orange
        ]
        self.GEM_HIGHLIGHT_COLORS = [(min(255, r+60), min(255, g+60), min(255, b+60)) for r,g,b in self.GEM_COLORS]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem_pos = None
        self.moves_remaining = 0
        self.gems_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # For animations
        self.animations = [] # List of animation objects/dicts
        self.particles = []

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Note: For full determinism, Python's `random` should also be seeded
            # if it's used anywhere. We use it for particles.
            np.random.seed(seed)
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_pos = None
        self.animations = []
        self.particles = []
        
        self._create_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Handle Input ---
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1

        if space_pressed:
            # Sound placeholder: pygame.mixer.Sound("select.wav").play()
            if self.selected_gem_pos is None:
                self.selected_gem_pos = list(self.cursor_pos)
            else:
                if self._is_adjacent(self.selected_gem_pos, self.cursor_pos):
                    reward = self._attempt_swap(self.selected_gem_pos, self.cursor_pos)
                self.selected_gem_pos = None # Deselect after any swap attempt

        # --- Check Termination Conditions ---
        if self.gems_collected >= self.GEM_GOAL:
            reward += 100
            terminated = True
            self.game_over = True
            # Sound placeholder: pygame.mixer.Sound("win.wav").play()
        elif self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
            # Sound placeholder: pygame.mixer.Sound("lose.wav").play()
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # This function renders the current static state.
        # Animations are handled and rendered inside the step logic for visual effect.
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "moves_remaining": self.moves_remaining,
        }

    # --- Core Game Logic ---

    def _create_initial_grid(self):
        self.grid = np.random.randint(0, self.NUM_GEM_TYPES, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        while True:
            matches = self._find_all_matches()
            if not matches:
                if self._find_possible_moves():
                    break
                else: # No initial matches, but also no possible moves. Reshuffle.
                    np.random.shuffle(self.grid.flat)
            else:
                # Remove initial matches without awarding points
                self._remove_gems(matches)
                self._apply_gravity()
                self._fill_top_rows()

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _swap_gems(self, pos1, pos2):
        self.grid[pos1[0], pos1[1]], self.grid[pos2[0], pos2[1]] = \
            self.grid[pos2[0], pos2[1]], self.grid[pos1[0], pos1[1]]

    def _attempt_swap(self, pos1, pos2):
        # Sound placeholder: pygame.mixer.Sound("swap.wav").play()
        self._animate_swap(pos1, pos2)
        self._swap_gems(pos1, pos2)
        
        matches = self._find_all_matches()
        if not matches:
            # Invalid move, swap back
            # Sound placeholder: pygame.mixer.Sound("invalid_swap.wav").play()
            self._animate_swap(pos1, pos2) # Animate the swap back
            self._swap_gems(pos1, pos2)
            return 0
        
        # Valid move
        self.moves_remaining -= 1
        total_reward = 0
        
        # --- Cascade Loop ---
        while matches:
            num_matched = len(matches)
            
            # Calculate reward
            match_reward = num_matched
            
            # Check for large matches (4 or 5)
            all_coords = set(matches)
            rows = {}
            cols = {}
            for x,y in all_coords:
                rows.setdefault(y, []).append(x)
                cols.setdefault(x, []).append(y)
            
            for r in rows.values():
                if len(r) >= 4: match_reward += 5
            for c in cols.values():
                if len(c) >= 4: match_reward += 5

            total_reward += match_reward
            self.score += match_reward
            self.gems_collected += num_matched
            
            # Animate gems popping
            self._animate_pop(matches)

            # Remove gems and let new ones fall
            self._remove_gems(matches)
            falling_gems = self._apply_gravity()
            new_gems = self._fill_top_rows()
            
            # Animate falling gems
            self._animate_fall(falling_gems, new_gems)

            # Check for new matches
            matches = self._find_all_matches()
        
        # Ensure board is not stuck after a move
        if not self.game_over and not self._find_possible_moves():
            self._reshuffle_board()

        return total_reward

    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Horizontal check
                if x < self.GRID_WIDTH - 2:
                    if self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y] and self.grid[x,y] != -1:
                        matches.add((x, y)); matches.add((x+1, y)); matches.add((x+2, y))
                # Vertical check
                if y < self.GRID_HEIGHT - 2:
                    if self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2] and self.grid[x,y] != -1:
                        matches.add((x, y)); matches.add((x, y+1)); matches.add((x, y+2))
        return list(matches)

    def _remove_gems(self, matches):
        for x, y in matches:
            self.grid[x, y] = -1 # Mark as empty

    def _apply_gravity(self):
        falling_gems = {} # { (end_x, end_y): (start_x, start_y) }
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == -1:
                    empty_slots.append(y)
                elif empty_slots:
                    fall_to_y = empty_slots.pop(0)
                    self.grid[x, fall_to_y] = self.grid[x, y]
                    self.grid[x, y] = -1
                    empty_slots.append(y)
                    falling_gems[(x, fall_to_y)] = (x, y)
        return falling_gems
    
    def _fill_top_rows(self):
        new_gems = {} # { (x,y): gem_type }
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == -1:
                    self.grid[x, y] = np.random.randint(0, self.NUM_GEM_TYPES)
                    new_gems[(x,y)] = self.grid[x,y]
        return new_gems

    def _find_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self._swap_gems((x, y), (x + 1, y))
                    if self._find_all_matches():
                        self._swap_gems((x, y), (x + 1, y)) # Swap back
                        return True
                    self._swap_gems((x, y), (x + 1, y)) # Swap back
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_gems((x, y), (x, y + 1))
                    if self._find_all_matches():
                        self._swap_gems((x, y), (x, y + 1)) # Swap back
                        return True
                    self._swap_gems((x, y), (x, y + 1)) # Swap back
        return False
    
    def _reshuffle_board(self):
        # Sound placeholder: pygame.mixer.Sound("reshuffle.wav").play()
        flat_grid = self.grid.flatten()
        np.random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Animate a "shuffle" effect
        animation_duration = 30 # frames
        for i in range(animation_duration):
            self._render_frame(is_shuffling=True, shuffle_progress=i/animation_duration)
            self.clock.tick(60)
            
        # Ensure new board is valid
        while self._find_all_matches() or not self._find_possible_moves():
            np.random.shuffle(self.grid.flat)

    # --- Animation ---

    def _run_animation_loop(self, duration_frames):
        for _ in range(duration_frames):
            self._render_frame()
            self.clock.tick(60)

    def _animate_swap(self, pos1, pos2):
        duration = 15 # frames
        self.animations.append({'type': 'swap', 'pos1': pos1, 'pos2': pos2, 'progress': 0, 'duration': duration})
        self._run_animation_loop(duration)
        self.animations.clear()

    def _animate_pop(self, matches):
        # Sound placeholder: pygame.mixer.Sound("match.wav").play()
        duration = 20 # frames
        self.animations.append({'type': 'pop', 'matches': matches, 'progress': 0, 'duration': duration})
        for x,y in matches:
            gem_type = self.grid[x,y]
            if gem_type != -1:
                for _ in range(5):
                    self.particles.append(self._create_particle(x, y, gem_type))
        self._run_animation_loop(duration)
        self.animations.clear()

    def _animate_fall(self, falling_gems, new_gems):
        # Sound placeholder: pygame.mixer.Sound("fall.wav").play()
        duration = 20 # frames
        self.animations.append({'type': 'fall', 'falling': falling_gems, 'new': new_gems, 'progress': 0, 'duration': duration})
        self._run_animation_loop(duration)
        self.animations.clear()

    def _create_particle(self, grid_x, grid_y, gem_type):
        center_x, center_y = self._grid_to_pixel(grid_x, grid_y, center=True)
        return {
            'x': center_x, 'y': center_y,
            'vx': random.uniform(-2, 2), 'vy': random.uniform(-3, 1),
            'lifespan': 20, 'max_lifespan': 20,
            'color': self.GEM_COLORS[gem_type],
            'radius': random.uniform(2, 5)
        }

    # --- Rendering ---

    def _grid_to_pixel(self, x, y, center=False):
        px = self.GRID_OFFSET_X + x * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return px, py

    def _render_frame(self, is_shuffling=False, shuffle_progress=0.0):
        # Background
        self.screen.fill(self.COLOR_BG)
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, 
                                self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, y), self.GRID_LINE_WIDTH)
        
        # Process animations to get gem positions/scales for this frame
        render_info = {}
        for anim in self.animations:
            progress = min(1.0, anim['progress'] / anim['duration'])
            anim['progress'] += 1
            if anim['type'] == 'swap':
                p1, p2 = anim['pos1'], anim['pos2']
                render_info[tuple(p1)] = {'offset_pos': p2, 'interp': progress}
                render_info[tuple(p2)] = {'offset_pos': p1, 'interp': progress}
            elif anim['type'] == 'pop':
                for pos in anim['matches']:
                    render_info[tuple(pos)] = {'scale': 1.0 - progress, 'alpha': 255 * (1.0 - progress)}
            elif anim['type'] == 'fall':
                for end_pos, start_pos in anim['falling'].items():
                     render_info[tuple(end_pos)] = {'offset_pos': start_pos, 'interp': 1.0 - progress}
                for pos in anim['new'].keys():
                    start_pos = (pos[0], -1)
                    render_info[tuple(pos)] = {'offset_pos': start_pos, 'interp': 1.0 - progress}

        # Gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[x, y]
                if gem_type == -1: continue

                info = render_info.get((x, y), {})
                px, py = self._grid_to_pixel(x, y)
                scale = info.get('scale', 1.0)
                alpha = info.get('alpha', 255)

                if 'offset_pos' in info:
                    interp = info['interp']
                    off_x, off_y = info['offset_pos']
                    start_px, start_py = self._grid_to_pixel(off_x, off_y)
                    px = int(start_px + (px - start_px) * interp)
                    py = int(start_py + (py - start_py) * interp)

                if is_shuffling:
                    angle = (x + y) * 3.14 + shuffle_progress * 10
                    radius = (1.0 - shuffle_progress) * 20
                    px += int(math.cos(angle) * radius)
                    py += int(math.sin(angle) * radius)
                    scale = shuffle_progress * 0.5 + 0.5
                
                self._draw_gem(self.screen, px, py, gem_type, scale, alpha)

        # Particles
        self._update_and_draw_particles()

        # Cursor and selection
        if not self.game_over:
            self._render_cursor()
        
        # UI
        self._render_ui()
        
        # Game Over Screen
        if self.game_over:
            self._render_game_over()

    def _draw_gem(self, surface, px, py, gem_type, scale=1.0, alpha=255):
        if alpha <= 0: return
        size = int(self.CELL_SIZE * 0.8 * scale)
        if size <= 1: return
        
        center_x = px + self.CELL_SIZE // 2
        center_y = py + self.CELL_SIZE // 2

        color = self.GEM_COLORS[gem_type]
        highlight = self.GEM_HIGHLIGHT_COLORS[gem_type]

        if alpha < 255:
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            self._draw_gem_shape(temp_surf, 0, 0, size, color, highlight)
            temp_surf.set_alpha(alpha)
            surface.blit(temp_surf, (center_x - size//2, center_y - size//2))
        else:
            self._draw_gem_shape(surface, center_x - size//2, center_y - size//2, size, color, highlight)
            
    def _draw_gem_shape(self, surface, x, y, size, color, highlight_color):
        half = size // 2
        quarter = size // 4
        
        # Main diamond shape
        points = [(x + half, y), (x + size, y + half), (x + half, y + size), (x, y + half)]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

        # Highlight facet
        h_points = [(x + half, y), (x + size, y + half), (x + half, y + half)]
        pygame.gfxdraw.filled_polygon(surface, h_points, highlight_color)
        pygame.gfxdraw.aapolygon(surface, h_points, highlight_color)
        
        # Inner glare
        glare_points = [(x + half, y + quarter), (x + half + quarter, y + half), (x + half, y + half + quarter), (x + half - quarter, y + half)]
        pygame.gfxdraw.filled_polygon(surface, glare_points, (255, 255, 255, 90))

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # gravity
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), color)
                remaining_particles.append(p)
        self.particles = remaining_particles

    def _render_cursor(self):
        # Draw selection highlight
        if self.selected_gem_pos:
            px, py = self._grid_to_pixel(*self.selected_gem_pos)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTED)
            self.screen.blit(s, (px, py))
        
        # Draw cursor
        px, py = self._grid_to_pixel(*self.cursor_pos)
        pygame.draw.rect(self.screen, (255,255,255), (px, py, self.CELL_SIZE, self.CELL_SIZE), 3)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, x, y, center=False):
            text_surf = font.render(text, True, (0,0,0,128))
            if center:
                text_rect = text_surf.get_rect(center=(x+2, y+2))
            else:
                text_rect = text_surf.get_rect(topleft=(x+2, y+2))
            self.screen.blit(text_surf, text_rect)
            
            text_surf = font.render(text, True, color)
            if center:
                text_rect = text_surf.get_rect(center=(x, y))
            else:
                text_rect = text_surf.get_rect(topleft=(x, y))
            self.screen.blit(text_surf, text_rect)

        # Draw UI text
        draw_text(f"Score: {self.score}", self.font_medium, self.COLOR_TEXT, 10, 10)
        draw_text(f"Moves: {self.moves_remaining}", self.font_medium, self.COLOR_TEXT, self.SCREEN_WIDTH - 150, 10)
        draw_text(f"Gems: {self.gems_collected} / {self.GEM_GOAL}", self.font_medium, self.COLOR_TEXT, self.SCREEN_WIDTH/2, 10, center=True)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.gems_collected >= self.GEM_GOAL:
            msg = "YOU WIN!"
            color = (100, 255, 100)
        else:
            msg = "GAME OVER"
            color = (255, 100, 100)
            
        text_surf = self.font_large.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(text_surf, text_rect)

        final_score_surf = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
        self.screen.blit(final_score_surf, final_score_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up the display window
    pygame.display.set_caption("Gem Swap")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        # --- Human Input to Action Conversion ---
        movement = 0 # no-op
        space = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q: # Quit on 'q' key
                    running = False

        if not done:
            action = [movement, space, 0] # shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward > 0:
                print(f"Reward: {reward}, Score: {info['score']}, Moves Left: {info['moves_remaining']}")
            if done:
                print(f"Game Over! Final Info: {info}")

        # --- Update Display ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose
        # The observation is already transposed correctly for gym, so we need to undo it for pygame display
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(10) # Control human play speed

    env.close()