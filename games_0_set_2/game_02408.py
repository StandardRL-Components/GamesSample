# Generated: 2025-08-27T20:17:45.136536
# Source Brief: brief_02408.md
# Brief Index: 2408

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to move and push boxes. The goal is to get all brown boxes onto the green targets."

    # Must be a short, user-facing description of the game:
    game_description = "A side-scrolling puzzle game where you push boxes that are affected by gravity. Plan your moves carefully to solve the level."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (40, 50, 60)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_BOX = (180, 120, 80)
    COLOR_PLATFORM = (100, 110, 120)
    COLOR_TARGET = (80, 200, 120)
    COLOR_TARGET_FILLED = (150, 255, 180)
    COLOR_TEXT = (230, 240, 250)

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
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables to be defined in reset()
        self.player_pos = None
        self.boxes = None
        self.platforms = None
        self.targets = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.particles = []
        self.prev_on_target_count = 0
        
        # The validation function is called to ensure the implementation is correct.
        # It needs a fully initialized state, which is done by reset().
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        # --- Level Design ---
        self.player_pos = [1, 8]
        # Using lists for positions to make them mutable
        self.boxes = [
            [4, 4], [6, 4], [8, 4], [10, 4], [12, 4]
        ]
        # Using sets for static obstacles for fast lookups
        platform_coords = [
            (i, 9) for i in range(self.GRID_WIDTH) # Floor
        ] + [
            (i, 5) for i in range(3, 13)
        ] + [
            (i, 3) for i in range(7, 9)
        ] + [
            (0, i) for i in range(self.GRID_HEIGHT) # Left Wall
        ] + [
            (self.GRID_WIDTH - 1, i) for i in range(self.GRID_HEIGHT) # Right Wall
        ]
        self.platforms = set(tuple(p) for p in platform_coords)

        self.targets = {
            (3, 8), (5, 8), (7, 8), (9, 8), (11, 8)
        }
        
        self._apply_gravity()
        self.prev_on_target_count = self._count_boxes_on_target()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.1  # Cost for taking a step
        self.steps += 1

        # --- Player Movement and Pushing Logic ---
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            next_player_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            box_idx = self._get_box_at(next_player_pos)
            if box_idx is not None:
                next_box_pos = [next_player_pos[0] + dx, next_player_pos[1] + dy]
                obstacles = self.platforms.union(self._get_box_coords(exclude_idx=box_idx))
                if tuple(next_box_pos) not in obstacles:
                    self.boxes[box_idx] = next_box_pos
                    self.player_pos = next_player_pos
                    # sfx: box_push
            else:
                obstacles = self.platforms.union(self._get_box_coords())
                if tuple(next_player_pos) not in obstacles:
                    self.player_pos = next_player_pos
                    # sfx: player_step

        self._apply_gravity()

        # --- Calculate Rewards and Check Game State ---
        for box in self.boxes:
            if box[1] >= self.GRID_HEIGHT:
                self.game_over = True
                reward += -100
                # sfx: game_over_sound
                break
        
        if not self.game_over:
            current_on_target_count = self._count_boxes_on_target()
            newly_on_target = current_on_target_count - self.prev_on_target_count
            if newly_on_target > 0:
                reward += newly_on_target * 10
                # sfx: success_chime
                for box_pos in self.boxes:
                    if tuple(box_pos) in self.targets:
                         self._spawn_particles(box_pos)

            self.prev_on_target_count = current_on_target_count
            
            if current_on_target_count == len(self.boxes):
                self.game_over = True
                reward += 100
                # sfx: level_complete_fanfare
        
        self.score += reward
        terminated = self.game_over or self.steps >= 1000
        if self.steps >= 1000 and not self.game_over:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_gravity(self):
        moved_in_pass = True
        while moved_in_pass:
            moved_in_pass = False
            sorted_indices = sorted(range(len(self.boxes)), key=lambda k: self.boxes[k][1], reverse=True)
            
            for i in sorted_indices:
                box = self.boxes[i]
                below_pos = (box[0], box[1] + 1)
                
                obstacles = self.platforms.union(self._get_box_coords(exclude_idx=i))
                
                if below_pos not in obstacles and below_pos[1] < self.GRID_HEIGHT:
                    self.boxes[i][1] += 1
                    moved_in_pass = True
                    # sfx: box_thud

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_platforms_and_targets()
        self._draw_boxes()
        self._draw_player()
        self._update_and_draw_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "boxes_on_target": self._count_boxes_on_target(),
        }

    def _get_box_at(self, pos):
        try:
            return self.boxes.index(list(pos))
        except ValueError:
            return None

    def _get_box_coords(self, exclude_idx=-1):
        return {tuple(box) for i, box in enumerate(self.boxes) if i != exclude_idx}

    def _count_boxes_on_target(self):
        if not self.boxes or not self.targets:
            return 0
        return sum(1 for box_pos in self.boxes if tuple(box_pos) in self.targets)

    def _spawn_particles(self, grid_pos):
        # sfx: sparkle_sound
        pixel_pos_x = (grid_pos[0] + 0.5) * self.GRID_SIZE
        pixel_pos_y = (grid_pos[1] + 0.5) * self.GRID_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [pixel_pos_x, pixel_pos_y], 'vel': vel, 'life': life,
                'max_life': life, 'color': self.COLOR_TARGET_FILLED
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # particle gravity
            p['life'] -= 1
            
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (*p['color'], alpha), (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_platforms_and_targets(self):
        for x, y in self.platforms:
            if (x, y) not in self.targets:
                rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, rect)
        
        for x, y in self.targets:
            rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            is_filled = tuple([x, y]) in [tuple(b) for b in self.boxes]
            color = self.COLOR_TARGET_FILLED if is_filled else self.COLOR_TARGET
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            border_color = tuple(c * 0.8 for c in color)
            pygame.draw.rect(self.screen, border_color, rect, width=3, border_radius=4)

    def _draw_boxes(self):
        for x, y in self.boxes:
            rect = pygame.Rect(x * self.GRID_SIZE + 4, y * self.GRID_SIZE + 4, self.GRID_SIZE - 8, self.GRID_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_BOX, rect, border_radius=6)
            border_color = tuple(c * 0.8 for c in self.COLOR_BOX)
            pygame.draw.rect(self.screen, border_color, rect, width=3, border_radius=6)

    def _draw_player(self):
        center_x = int((self.player_pos[0] + 0.5) * self.GRID_SIZE)
        center_y = int((self.player_pos[1] + 0.5) * self.GRID_SIZE)
        radius = self.GRID_SIZE // 2 - 4
        
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        moves_text = self.font.render(f"Moves: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            win = self._count_boxes_on_target() == len(self.boxes)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "LEVEL COMPLETE!" if win else "GAME OVER"
            end_color = self.COLOR_TARGET_FILLED if win else (255, 100, 100)
            end_text = self.font.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space (a static property)
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset, which initializes the environment state
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")