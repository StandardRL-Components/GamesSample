
# Generated: 2025-08-27T19:38:12.665447
# Source Brief: brief_02210.md
# Brief Index: 2210

        
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
        "Controls: Arrow keys to move cursor. Space to swap gems. "
        "Swap the selected gem with an adjacent one in the direction you last moved."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match-3 puzzle game. Align 3 or more gems of the same color to collect them. "
        "Collect 100 gems before the 60-second timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    GEM_TYPES = 5
    
    GAME_TIMER_SECONDS = 60
    WIN_GEM_COUNT = 100

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Sizing
    GRID_AREA_WIDTH = 320
    GRID_AREA_HEIGHT = 320
    GRID_TOP_LEFT_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_TOP_LEFT_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2
    CELL_SIZE = GRID_AREA_WIDTH // GRID_WIDTH
    GEM_RADIUS = int(CELL_SIZE * 0.4)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.last_move_direction = None
        self.prev_space_held = None
        self.steps = None
        self.score = None
        self.gems_collected = None
        self.game_timer = None
        self.game_over = None
        self.particles = []
        self.match_animations = []
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def _initialize_grid(self):
        grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                possible_gems = list(range(self.GEM_TYPES))
                # Avoid creating matches on spawn
                if x > 1 and grid[y, x-1] == grid[y, x-2]:
                    if grid[y, x-1] in possible_gems:
                        possible_gems.remove(grid[y, x-1])
                if y > 1 and grid[y-1, x] == grid[y-2, x]:
                    if grid[y-1, x] in possible_gems:
                        possible_gems.remove(grid[y-1, x])
                grid[y, x] = self.np_random.choice(possible_gems)
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_timer = self.GAME_TIMER_SECONDS
        self.game_over = False

        self.grid = self._initialize_grid()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_direction = (0, 0) # No initial move direction
        self.prev_space_held = False

        self.particles = []
        self.match_animations = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.game_timer -= 1 / self.FPS

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle player input
        moved = self._handle_movement(movement)
        swap_attempted, successful_swap = self._handle_swap(space_held)

        if swap_attempted and not successful_swap:
            reward -= 0.2 # Penalty for invalid swap

        # 2. Process match-fall-match cascade
        cascade_reward = self._process_cascades()
        reward += cascade_reward

        # 3. Update animations
        self._update_animations()

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.gems_collected >= self.WIN_GEM_COUNT:
                reward += 100 # Win bonus
            elif self.game_timer <= 0:
                reward += -50 # Lose penalty
        
        self.prev_space_held = space_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        moved = False
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        if dx != 0 or dy != 0:
            # Grid wrap-around logic
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
            self.last_move_direction = (dx, dy)
            moved = True
        return moved

    def _handle_swap(self, space_held):
        swap_attempted = False
        successful_swap = False
        
        is_space_press = space_held and not self.prev_space_held
        if is_space_press and self.last_move_direction != (0, 0):
            swap_attempted = True
            cx, cy = self.cursor_pos
            dx, dy = self.last_move_direction
            nx, ny = cx + dx, cy + dy
            
            # Check if neighbor is within bounds
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                # Perform swap
                self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
                
                # Check if swap results in a match
                matches = self._find_matches()
                if not matches:
                    # If not, swap back
                    self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
                else:
                    successful_swap = True
        return swap_attempted, successful_swap

    def _process_cascades(self):
        total_cascade_reward = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            match_reward = self._process_matches(matches)
            total_cascade_reward += match_reward
            self._apply_gravity_and_refill()
        return total_cascade_reward

    def _find_matches(self):
        matched_coords = set()
        # Horizontal matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[y, x] == self.grid[y, x+1] == self.grid[y, x+2] and self.grid[y,x] != -1:
                    match_len = 3
                    while x + match_len < self.GRID_WIDTH and self.grid[y, x] == self.grid[y, x + match_len]:
                        match_len += 1
                    for i in range(match_len):
                        matched_coords.add((x+i, y))
        # Vertical matches
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[y, x] == self.grid[y+1, x] == self.grid[y+2, x] and self.grid[y,x] != -1:
                    match_len = 3
                    while y + match_len < self.GRID_HEIGHT and self.grid[y, x] == self.grid[y + match_len, x]:
                        match_len += 1
                    for i in range(match_len):
                        matched_coords.add((x, y+i))
        return matched_coords

    def _process_matches(self, matched_coords):
        reward = 0
        
        # Group connected components to reward larger matches
        groups = self._group_matches(matched_coords)
        
        for group in groups:
            reward += len(group)
            self.gems_collected += len(group)
            self.score += len(group) * 10
            
            if len(group) >= 4:
                reward += 5
                self.score += 50 # Bonus score
            
            for x, y in group:
                gem_type = self.grid[y, x]
                if gem_type != -1:
                    self._create_particles(x, y, gem_type)
                    self.match_animations.append({'coords': (x, y), 'timer': 15, 'color': self.GEM_COLORS[gem_type]})
                    self.grid[y, x] = -1 # Mark for removal
        return reward
    
    def _group_matches(self, coords):
        if not coords:
            return []
        
        groups = []
        coords_list = list(coords)
        
        while coords_list:
            group = set()
            queue = [coords_list.pop(0)]
            group.add(queue[0])
            
            head = 0
            while head < len(queue):
                x, y = queue[head]
                head += 1
                
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in coords_list:
                        group.add(neighbor)
                        queue.append(neighbor)
                        coords_list.remove(neighbor)
            groups.append(group)
        return groups

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] == -1:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[y + empty_slots, x] = self.grid[y, x]
                    self.grid[y, x] = -1
            
            for y in range(empty_slots):
                self.grid[y, x] = self.np_random.integers(0, self.GEM_TYPES)

    def _create_particles(self, grid_x, grid_y, gem_type):
        # Sound: Gem Match
        px, py = self._grid_to_pixel(grid_x, grid_y)
        color = self.GEM_COLORS[gem_type]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color, 'radius': radius})

    def _update_animations(self):
        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['vel'][1] += 0.1 # Gravity
        
        # Update match animations
        self.match_animations = [m for m in self.match_animations if m['timer'] > 0]
        for m in self.match_animations:
            m['timer'] -= 1

    def _check_termination(self):
        return self.gems_collected >= self.WIN_GEM_COUNT or self.game_timer <= 0

    def _grid_to_pixel(self, x, y):
        px = self.GRID_TOP_LEFT_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_TOP_LEFT_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=8)

        # Draw gems
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem_type = self.grid[y, x]
                if gem_type != -1:
                    px, py = self._grid_to_pixel(x, y)
                    color = self.GEM_COLORS[gem_type]
                    
                    is_animating = False
                    for anim in self.match_animations:
                        if anim['coords'] == (x, y):
                            # Flash effect
                            if (anim['timer'] // 2) % 2 == 0:
                                color = (255, 255, 255)
                            is_animating = True
                            break
                    
                    if not is_animating:
                        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), self.GEM_RADIUS, color)
                        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), self.GEM_RADIUS, color)
                        # Highlight
                        highlight_color = (min(255, c+60) for c in color)
                        pygame.gfxdraw.filled_circle(self.screen, int(px - self.GEM_RADIUS*0.3), int(py - self.GEM_RADIUS*0.3), self.GEM_RADIUS//3, tuple(highlight_color))

        # Draw particles
        for p in self.particles:
            life_ratio = p['lifespan'] / p['max_life']
            current_radius = int(p['radius'] * life_ratio)
            if current_radius > 0:
                color = tuple(int(c * life_ratio) for c in p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), current_radius, color)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_TOP_LEFT_X + cx * self.CELL_SIZE,
            self.GRID_TOP_LEFT_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=5)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, center=True):
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surf = font.render(text, True, color)
            
            shadow_rect = shadow_surf.get_rect()
            text_rect = text_surf.get_rect()

            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
                text_rect.center = pos
            else: # Align left
                shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
                text_rect.topleft = pos

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

        # Gem Count (Top Left)
        gem_icon_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(gem_icon_surf, 15, 15, 12, self.GEM_COLORS[0])
        pygame.gfxdraw.aacircle(gem_icon_surf, 15, 15, 12, self.GEM_COLORS[0])
        self.screen.blit(gem_icon_surf, (20, 15))
        gem_text = f"{self.gems_collected} / {self.WIN_GEM_COUNT}"
        draw_text(gem_text, self.font_medium, self.COLOR_TEXT, (60, 30), center=False)

        # Timer (Top Right)
        timer_text = f"Time: {max(0, math.ceil(self.game_timer))}"
        draw_text(timer_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - 100, 30))

        # Score (Bottom Center)
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.gems_collected >= self.WIN_GEM_COUNT:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "TIME'S UP!"
                color = (255, 100, 100)
            draw_text(msg, self.font_large, color, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "time_remaining": self.game_timer,
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
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Grid")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track held keys for smooth movement
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False
    }
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # --- Action Mapping ---
        movement = 0 # none
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space_btn = 1 if keys_held[pygame.K_SPACE] else 0
        shift_btn = 1 if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT] else 0
        
        action = [movement, space_btn, shift_btn]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Termination ---
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        env.clock.tick(GameEnv.FPS)

    env.close()