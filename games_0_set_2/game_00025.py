import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select/deselect a gem. "
        "Use arrow keys again to swap with an adjacent gem."
    )

    game_description = (
        "Swap adjacent gems to create matches of three or more. Create cascading combos to "
        "multiply your score and reach the target before you run out of moves."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    NUM_GEM_TYPES = 5
    WIN_SCORE = 1000
    STARTING_MOVES = 20
    
    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_UI_BG = (30, 35, 60)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_SELECTED = (255, 255, 255, 200)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    # --- Game Settings ---
    ANIMATION_SPEED = 0.2  # Progress per frame, so 1/0.2 = 5 frames for an animation

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
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)

        self.grid_start_x = (self.SCREEN_WIDTH - self.SCREEN_HEIGHT) // 2
        self.grid_start_y = 0
        self.cell_size = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.gem_size = int(self.cell_size * 0.8)

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.steps = 0
        
        self.previous_action = self.action_space.sample()
        self.game_state = "IDLE" # IDLE, SWAPPING, RESOLVING, GAME_OVER
        self.animations = []
        self.particles = []
        self.reward_queue = []
        self.chain_multiplier = 1

        # self.validate_implementation() # This is for internal testing, not part of the final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.STARTING_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.game_state = "IDLE"
        self.animations = []
        self.particles = []
        self.reward_queue = []
        self.chain_multiplier = 1

        self._create_and_validate_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        self.steps += 1
        reward = 0
        
        # Process queued rewards from previous logic steps
        reward += sum(self.reward_queue)
        self.reward_queue = []

        if self.game_state != "GAME_OVER":
            self._handle_animations()

            if self.game_state == "IDLE":
                self._handle_input(action)

        self.previous_action = action

        terminated = self.game_over
        if terminated and self.game_state != "GAME_OVER": # First frame of game over
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward += -10 # Loss penalty
            self.game_state = "GAME_OVER"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prev_movement, prev_space_held, _ = self.previous_action[0], self.previous_action[1] == 1, self.previous_action[2] == 1

        space_pressed = space_held and not prev_space_held
        movement_pressed = movement != 0 and movement != prev_movement

        # --- Cursor Movement ---
        if self.selected_pos is None and movement_pressed:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # --- Gem Selection ---
        if space_pressed:
            if self.selected_pos is None:
                self.selected_pos = list(self.cursor_pos)
            elif self.selected_pos == self.cursor_pos:
                self.selected_pos = None
            else: # Select a new gem
                self.selected_pos = list(self.cursor_pos)

        # --- Gem Swapping ---
        if self.selected_pos is not None and movement_pressed:
            target_pos = list(self.selected_pos)
            if movement == 1: target_pos[1] -= 1
            elif movement == 2: target_pos[1] += 1
            elif movement == 3: target_pos[0] -= 1
            elif movement == 4: target_pos[0] += 1

            if 0 <= target_pos[0] < self.GRID_SIZE and 0 <= target_pos[1] < self.GRID_SIZE:
                self._attempt_swap(self.selected_pos, target_pos)

    def _attempt_swap(self, pos1, pos2):
        self.moves_remaining -= 1
        self.selected_pos = None
        
        self.grid[pos1[1], pos1[0]], self.grid[pos2[1], pos2[0]] = self.grid[pos2[1], pos2[0]], self.grid[pos1[1], pos1[0]]
        
        matches1 = self._find_matches_at_pos(pos1)
        matches2 = self._find_matches_at_pos(pos2)
        
        if not matches1 and not matches2:
            self.animations.append({
                "type": "swap", "pos1": pos1, "pos2": pos2, "progress": 0.0, "callback": self._swap_back
            })
            self.reward_queue.append(-0.1)
        else:
            self.chain_multiplier = 1
            self.animations.append({
                "type": "swap", "pos1": pos1, "pos2": pos2, "progress": 0.0, "callback": self._resolve_matches
            })
        
        self.game_state = "SWAPPING"

    def _swap_back(self, data):
        pos1, pos2 = data['pos1'], data['pos2']
        self.grid[pos1[1], pos1[0]], self.grid[pos2[1], pos2[0]] = self.grid[pos2[1], pos2[0]], self.grid[pos1[1], pos1[0]]
        self.game_state = "IDLE"
        if self.moves_remaining <= 0:
            self.game_over = True

    def _resolve_matches(self, data=None):
        all_matches = self._find_all_matches()

        if not all_matches:
            self.game_state = "IDLE"
            if self.moves_remaining <= 0 or self.score >= self.WIN_SCORE:
                self.game_over = True
            if not self._has_possible_moves():
                self._create_and_validate_board()
            return

        num_cleared = len(all_matches)
        self.reward_queue.append(num_cleared)
        self.score += num_cleared * 10 * self.chain_multiplier
        if self.chain_multiplier > 1:
            self.reward_queue.append(5)

        for y, x in all_matches:
            self._create_particles(x, y, self.grid[y, x])
            self.grid[y, x] = -1

        self.animations.append({
            "type": "fall", "progress": 0.0, "callback": self._post_fall_check
        })
        self.game_state = "RESOLVING"
        self.chain_multiplier += 1

    def _post_fall_check(self, data=None):
        self._resolve_matches()

    def _handle_animations(self):
        if not self.animations:
            return

        anim = self.animations[0]
        anim['progress'] = min(1.0, anim['progress'] + self.ANIMATION_SPEED)

        if anim['type'] == 'fall' and anim['progress'] >= 1.0:
            self._apply_gravity_and_refill()

        if anim['progress'] >= 1.0:
            self.animations.pop(0)
            if anim['callback']:
                anim['callback'](anim)

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] != -1:
                    if y != empty_row:
                        self.grid[empty_row, x] = self.grid[y, x]
                        self.grid[y, x] = -1
                    empty_row -= 1
            for y in range(empty_row, -1, -1):
                self.grid[y, x] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(self.GRID_SIZE + 1):
            x = self.grid_start_x + i * self.cell_size
            y = self.grid_start_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_start_y), (x, self.grid_start_y + self.GRID_SIZE * self.cell_size))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_start_x, y), (self.grid_start_x + self.GRID_SIZE * self.cell_size, y))

        if self.grid is None:
            return

        grid_render = np.copy(self.grid)
        
        swapping_anims = [a for a in self.animations if a['type'] == 'swap']
        if swapping_anims:
            anim = swapping_anims[0]
            p1, p2 = anim['pos1'], anim['pos2']
            gem1_type = self.grid[p2[1], p2[0]]
            gem2_type = self.grid[p1[1], p1[0]]
            
            self._draw_gem_at_interp_pos(gem1_type, p1, p2, anim['progress'])
            self._draw_gem_at_interp_pos(gem2_type, p2, p1, anim['progress'])
            grid_render[p1[1], p1[0]] = -1
            grid_render[p2[1], p2[0]] = -1

        falling_anims = [a for a in self.animations if a['type'] == 'fall']
        if falling_anims:
            progress = falling_anims[0]['progress']
            
            for x in range(self.GRID_SIZE):
                write_y = self.GRID_SIZE - 1
                for y in range(self.GRID_SIZE - 1, -1, -1):
                    if self.grid[y, x] != -1:
                        fall_dist = write_y - y
                        draw_y = y + fall_dist * progress
                        self._draw_gem_at_pixel_pos(self.grid[y,x], x, draw_y)
                        write_y -= 1
            
            grid_render = np.full_like(self.grid, -1)

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                gem_type = grid_render[y, x]
                if gem_type != -1:
                    self._draw_gem(gem_type, x, y)
        
        self._update_and_draw_particles()

        if not self.game_over:
            cursor_rect = pygame.Rect(self.grid_start_x + self.cursor_pos[0] * self.cell_size, 
                                      self.grid_start_y + self.cursor_pos[1] * self.cell_size, 
                                      self.cell_size, self.cell_size)
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(self.COLOR_CURSOR)
            self.screen.blit(s, cursor_rect.topleft)

            if self.selected_pos is not None:
                selected_rect = pygame.Rect(self.grid_start_x + self.selected_pos[0] * self.cell_size, 
                                            self.grid_start_y + self.selected_pos[1] * self.cell_size, 
                                            self.cell_size, self.cell_size)
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                pygame.draw.rect(s, self.COLOR_SELECTED, (0, 0, self.cell_size, self.cell_size), 4)
                self.screen.blit(s, selected_rect.topleft)

    def _draw_gem(self, gem_type, x, y):
        center_x = self.grid_start_x + x * self.cell_size + self.cell_size // 2
        center_y = self.grid_start_y + y * self.cell_size + self.cell_size // 2
        radius = self.gem_size // 2
        color = self.GEM_COLORS[gem_type]
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 2, 200, 340, highlight_color)

    def _draw_gem_at_interp_pos(self, gem_type, pos1, pos2, progress):
        x1, y1 = pos1
        x2, y2 = pos2
        interp_x = x1 + (x2 - x1) * progress
        interp_y = y1 + (y2 - y1) * progress
        self._draw_gem_at_pixel_pos(gem_type, interp_x, interp_y)
    
    def _draw_gem_at_pixel_pos(self, gem_type, grid_x, grid_y):
        center_x = int(self.grid_start_x + grid_x * self.cell_size + self.cell_size // 2)
        center_y = int(self.grid_start_y + grid_y * self.cell_size + self.cell_size // 2)
        radius = self.gem_size // 2
        color = self.GEM_COLORS[gem_type]
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 2, 200, 340, highlight_color)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.grid_start_x, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.grid_start_x + self.GRID_SIZE * self.cell_size, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        score_text = self.font_ui.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_ui.render(str(self.score), True, self.COLOR_UI_VALUE)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 50))
        
        moves_text = self.font_ui.render("MOVES", True, self.COLOR_UI_TEXT)
        moves_val = self.font_ui.render(str(self.moves_remaining), True, self.COLOR_UI_VALUE)
        self.screen.blit(moves_text, (20, 120))
        self.screen.blit(moves_val, (20, 150))
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_game_over.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_game_over.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _create_and_validate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_all_matches() and self._has_possible_moves():
                break

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != -1:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != -1:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_matches_at_pos(self, pos):
        r, c = pos[1], pos[0]
        gem_type = self.grid[r, c]
        if gem_type == -1: return set()
        
        h_matches, v_matches = { (r,c) }, { (r,c) }

        for i in range(1, self.GRID_SIZE):
            if c - i >= 0 and self.grid[r, c-i] == gem_type: h_matches.add((r, c-i))
            else: break
        for i in range(1, self.GRID_SIZE):
            if c + i < self.GRID_SIZE and self.grid[r, c+i] == gem_type: h_matches.add((r, c+i))
            else: break
        
        for i in range(1, self.GRID_SIZE):
            if r - i >= 0 and self.grid[r-i, c] == gem_type: v_matches.add((r-i, c))
            else: break
        for i in range(1, self.GRID_SIZE):
            if r + i < self.GRID_SIZE and self.grid[r+i, c] == gem_type: v_matches.add((r+i, c))
            else: break
            
        matches = set()
        if len(h_matches) >= 3: matches.update(h_matches)
        if len(v_matches) >= 3: matches.update(v_matches)
        return matches

    def _has_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_at_pos((c, r)) or self._find_matches_at_pos((c+1, r)):
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_at_pos((c, r)) or self._find_matches_at_pos((c, r+1)):
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _create_particles(self, x, y, gem_type):
        center_x = self.grid_start_x + x * self.cell_size + self.cell_size // 2
        center_y = self.grid_start_y + y * self.cell_size + self.cell_size // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.uniform(15, 30)
            self.particles.append(Particle([center_x, center_y], vel, color, lifespan))

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)
            else:
                p.draw(self.screen)

class Particle:
    def __init__(self, pos, vel, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.color = color

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1
        self.lifespan -= 1

    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, screen):
        alpha = max(0, int(255 * (self.lifespan / self.initial_lifespan)))
        color = self.color + (alpha,)
        radius = int(3 * (self.lifespan / self.initial_lifespan))
        if radius > 0:
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (radius, radius), radius)
            screen.blit(s, (int(self.pos[0] - radius), int(self.pos[1] - radius)))