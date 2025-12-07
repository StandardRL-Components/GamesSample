
# Generated: 2025-08-28T02:10:42.830288
# Source Brief: brief_04366.md
# Brief Index: 4366

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. "
        "Press space to select a gem, then select an adjacent gem to swap."
    )

    game_description = (
        "Swap adjacent gems to create matches of 3 or more in a grid-based puzzle game to reach a target score."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_GEM_TYPES = 6
    GEM_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * GEM_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * GEM_SIZE) // 2

    # Animation speeds
    SWAP_SPEED = 0.25  # seconds
    FALL_SPEED = 0.15   # seconds per grid cell
    DESTROY_SPEED = 0.2 # seconds

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (50, 55, 70)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SCORE = (255, 215, 0)
    COLOR_MOVES = (100, 200, 255)
    COLOR_SELECTOR = (255, 255, 255)
    
    GEM_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.render_mode = render_mode

        self.grid = None
        self.selector_pos = None
        self.selected_gem_pos = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.steps = None
        self.last_space_state = False
        self.animations = deque()
        self.particles = []
        self.is_processing_turn = False
        self.pending_reward = 0
        self.cleared_colors = None
        
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_remaining = 20
        self.game_over = False
        self.steps = 0
        self.selector_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.selected_gem_pos = None
        self.last_space_state = False
        self.animations.clear()
        self.particles.clear()
        self.is_processing_turn = False
        self.pending_reward = 0
        self.cleared_colors = set()

        self._generate_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = self.pending_reward
        self.pending_reward = 0

        if not self.game_over:
            if self.is_processing_turn:
                self._process_animations()
            else:
                self._handle_player_input(action)
        
        # Check for game over after processing
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= 500:
                reward += 100 # Win bonus
            else:
                reward += -10 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_player_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_state
        self.last_space_state = space_held

        # --- Movement ---
        if movement == 1: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 2: self.selector_pos[1] = min(self.GRID_HEIGHT - 1, self.selector_pos[1] + 1)
        elif movement == 3: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 4: self.selector_pos[0] = min(self.GRID_WIDTH - 1, self.selector_pos[0] + 1)

        # --- Selection/Swap ---
        if space_press:
            if self.selected_gem_pos is None:
                # First selection
                self.selected_gem_pos = self.selector_pos.copy()
                # sfx: select_gem.wav
            else:
                # Second selection (attempt swap)
                dist = np.sum(np.abs(self.selector_pos - self.selected_gem_pos))
                if dist == 1: # Is adjacent
                    self._attempt_swap(self.selected_gem_pos, self.selector_pos)
                self.selected_gem_pos = None # Deselect after any second click
                # sfx: deselect.wav or swap_fail.wav

    def _attempt_swap(self, pos1, pos2):
        self.moves_remaining -= 1
        
        # Perform swap on a temporary grid
        temp_grid = self.grid.copy()
        temp_grid[pos1[1], pos1[0]], temp_grid[pos2[1], pos2[0]] = temp_grid[pos2[1], pos2[0]], temp_grid[pos1[1], pos1[0]]
        
        matches1 = self._find_matches_at(pos1[1], pos1[0], temp_grid)
        matches2 = self._find_matches_at(pos2[1], pos2[0], temp_grid)
        all_matches = matches1.union(matches2)

        if all_matches:
            # Valid swap
            self.grid = temp_grid
            self.animations.append({
                "type": "swap", "pos1": pos1, "pos2": pos2, "progress": 0, "duration": self.SWAP_SPEED * 30
            })
            self.is_processing_turn = True
            self._start_match_process(all_matches)
            # sfx: swap_success.wav
        else:
            # Invalid swap
            self.pending_reward = -0.1
            self.animations.append({
                "type": "invalid_swap", "pos1": pos1, "pos2": pos2, "progress": 0, "duration": self.SWAP_SPEED * 30
            })
            self.is_processing_turn = True
            # sfx: swap_fail.wav

    def _process_animations(self):
        if not self.animations:
            self.is_processing_turn = False
            self._start_match_process(self._find_all_matches())
            return

        # Animate one frame
        for anim in list(self.animations):
            anim["progress"] += 1
            if anim["progress"] >= anim["duration"]:
                self.animations.remove(anim)

    def _start_match_process(self, matches):
        if not matches:
            return

        self.is_processing_turn = True
        
        # Reward and score
        num_gems_removed = len(matches)
        self.pending_reward += num_gems_removed
        self.score += num_gems_removed

        # sfx: match_found.wav
        for r, c in matches:
            self._create_particles(c, r, self.grid[r, c])
            self.animations.append({
                "type": "destroy", "pos": (c, r), "progress": 0, "duration": self.DESTROY_SPEED * 30
            })
            self.grid[r, c] = -1 # Mark as empty

        # Check for cleared color bonus
        current_gems = set(self.grid.flatten()) - {-1}
        all_gem_types = set(range(self.NUM_GEM_TYPES))
        newly_cleared = (all_gem_types - current_gems) - self.cleared_colors
        for color_type in newly_cleared:
            self.pending_reward += 10
            self.cleared_colors.add(color_type)
            # sfx: color_clear_bonus.wav

        # Schedule gravity after destruction animation
        self.animations.append({
            "type": "gravity", "progress": 0, "duration": self.DESTROY_SPEED * 30 + 1
        })

    def _apply_gravity_and_refill(self):
        max_fall_dist = 0
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        # Move gem down
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                        fall_dist = empty_row - r
                        max_fall_dist = max(max_fall_dist, fall_dist)
                        self.animations.append({
                            "type": "fall", "start_pos": (c, r), "end_pos": (c, empty_row),
                            "progress": 0, "duration": self.FALL_SPEED * fall_dist * 30
                        })
                    empty_row -= 1
        
        # Refill from top
        for c in range(self.GRID_WIDTH):
            fall_dist_offset = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)
                    fall_dist = r + 1 + fall_dist_offset
                    max_fall_dist = max(max_fall_dist, fall_dist)
                    self.animations.append({
                        "type": "fall", "start_pos": (c, -1 - fall_dist_offset), "end_pos": (c, r),
                        "progress": 0, "duration": self.FALL_SPEED * fall_dist * 30
                    })
                    fall_dist_offset += 1
        
        # Schedule next check after falling animations
        if max_fall_dist > 0:
            self.is_processing_turn = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.GEM_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.GEM_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.GEM_SIZE, y))

        # Prepare a set of animated gem positions to avoid double-drawing
        animated_positions = set()
        for anim in self.animations:
            if anim['type'] in ['swap', 'invalid_swap']:
                animated_positions.add(tuple(anim['pos1']))
                animated_positions.add(tuple(anim['pos2']))
            elif anim['type'] in ['destroy', 'fall']:
                pos = anim.get('pos') or anim.get('end_pos')
                animated_positions.add(tuple(pos)[::-1]) # (c,r) -> (r,c)

        # Draw static gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != -1 and (r, c) not in animated_positions:
                    self._draw_gem(c, r, self.grid[r, c])

        # Draw animated elements
        self._render_animations()
        self._render_particles()

        # Draw selector
        sel_x, sel_y = self.selector_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + sel_x * self.GEM_SIZE, self.GRID_OFFSET_Y + sel_y * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 3)

        # Draw selected gem highlight
        if self.selected_gem_pos is not None:
            sel_x, sel_y = self.selected_gem_pos
            rect = pygame.Rect(self.GRID_OFFSET_X + sel_x * self.GEM_SIZE, self.GRID_OFFSET_Y + sel_y * self.GEM_SIZE, self.GEM_SIZE, self.GEM_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 0), rect, 3)
            
    def _render_animations(self):
        for anim in list(self.animations):
            progress_ratio = anim['progress'] / anim['duration']
            
            if anim['type'] == 'swap' or anim['type'] == 'invalid_swap':
                c1, r1 = anim['pos1']
                c2, r2 = anim['pos2']
                gem_type1, gem_type2 = self.grid[r2, c2], self.grid[r1, c1]
                if anim['type'] == 'invalid_swap':
                    progress_ratio = 0.5 - abs(progress_ratio - 0.5) # go and come back
                
                x1, y1 = self._interp(c1, r1, c2, r2, progress_ratio)
                x2, y2 = self._interp(c2, r2, c1, r1, progress_ratio)
                self._draw_gem(x1, y1, gem_type1)
                self._draw_gem(x2, y2, gem_type2)

            elif anim['type'] == 'destroy':
                c, r = anim['pos']
                size_ratio = 1.0 - progress_ratio
                self._draw_gem(c, r, -1, size_ratio) # -1 gem type to use a placeholder color

            elif anim['type'] == 'fall':
                c1, r1 = anim['start_pos']
                c2, r2 = anim['end_pos']
                gem_type = self.grid[r2, c2]
                x, y = self._interp(c1, r1, c2, r2, progress_ratio)
                self._draw_gem(x, y, gem_type)
            
            elif anim['type'] == 'gravity' and anim['progress'] >= anim['duration']:
                self._apply_gravity_and_refill()

    def _render_particles(self):
        for p in list(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(0, p['size'] * (p['life'] / p['max_life']))
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], int(size))

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_MOVES)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text_str = "You Win!" if self.score >= 500 else "Game Over"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "is_animating": self.is_processing_turn,
        }

    def _check_termination(self):
        return self.moves_remaining <= 0 or self.score >= 500

    def _generate_initial_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_all_matches(self):
        all_matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                all_matches.update(self._find_matches_at(r, c, self.grid))
        return all_matches

    def _find_matches_at(self, r, c, grid):
        gem_type = grid[r, c]
        if gem_type == -1: return set()

        matches = set()
        # Horizontal
        h_matches = {(r, c)}
        for i in range(c - 1, -1, -1):
            if grid[r, i] == gem_type: h_matches.add((r, i))
            else: break
        for i in range(c + 1, self.GRID_WIDTH):
            if grid[r, i] == gem_type: h_matches.add((r, i))
            else: break
        if len(h_matches) >= 3: matches.update(h_matches)

        # Vertical
        v_matches = {(r, c)}
        for i in range(r - 1, -1, -1):
            if grid[i, c] == gem_type: v_matches.add((i, c))
            else: break
        for i in range(r + 1, self.GRID_HEIGHT):
            if grid[i, c] == gem_type: v_matches.add((i, c))
            else: break
        if len(v_matches) >= 3: matches.update(v_matches)
        
        return matches

    def _draw_gem(self, c, r, gem_type, scale=1.0):
        if gem_type < 0 or gem_type >= self.NUM_GEM_TYPES:
            return
            
        center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE / 2
        center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE / 2
        
        radius = int(self.GEM_SIZE * 0.4 * scale)
        if radius <= 0: return
        
        color = self.GEM_COLORS[gem_type]
        light_color = tuple(min(255, x + 50) for x in color)

        # Use different shapes for better accessibility
        if gem_type == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), radius, color)
            pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), int(radius*0.6), light_color)
        elif gem_type == 1: # Square
            rect = pygame.Rect(center_x - radius, center_y - radius, radius*2, radius*2)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, light_color, rect, 2)
        elif gem_type == 2: # Diamond
            points = [(center_x, center_y - radius), (center_x + radius, center_y),
                      (center_x, center_y + radius), (center_x - radius, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 3: # Triangle Up
            points = [(center_x, center_y - radius), (center_x + radius, center_y + radius*0.8),
                      (center_x - radius, center_y + radius*0.8)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Hexagon
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Cross
            rect1 = pygame.Rect(center_x - radius, center_y - radius*0.3, radius*2, radius*0.6)
            rect2 = pygame.Rect(center_x - radius*0.3, center_y - radius, radius*0.6, radius*2)
            pygame.draw.rect(self.screen, color, rect1)
            pygame.draw.rect(self.screen, color, rect2)

    def _interp(self, v1_c, v1_r, v2_c, v2_r, ratio):
        return v1_c + (v2_c - v1_c) * ratio, v1_r + (v2_r - v1_r) * ratio

    def _create_particles(self, c, r, gem_type):
        if gem_type < 0: return
        center_x = self.GRID_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE / 2
        center_y = self.GRID_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE / 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': random.randint(15, 30),
                'max_life': 30,
                'size': random.uniform(2, 5)
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Action mapping for human keyboard
        movement, space, shift = 0, 0, 0
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
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()