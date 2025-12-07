
# Generated: 2025-08-27T19:59:58.851324
# Source Brief: brief_02318.md
# Brief Index: 2318

        
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

    user_guide = (
        "Controls: Use arrow keys to move. Push all brown crates onto the red targets before time runs out."
    )

    game_description = (
        "An isometric puzzle game where you must push crates onto target locations against the clock. Plan your moves carefully to avoid getting crates stuck!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = 1800  # 60 seconds * 30 FPS

    # --- Colors ---
    COLOR_BG = (30, 35, 40)
    COLOR_FLOOR = (70, 80, 90)
    COLOR_WALL = (110, 120, 130)
    COLOR_WALL_TOP = (140, 150, 160)
    COLOR_PLAYER = (100, 220, 100)
    COLOR_CRATE = (180, 130, 90)
    COLOR_TARGET = (200, 80, 80)
    COLOR_TARGET_ACTIVE = (120, 255, 120)
    COLOR_SHADOW = (20, 25, 30, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 180, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)
    
    # --- Isometric Grid ---
    GRID_SIZE_X, GRID_SIZE_Y = 10, 10
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    BLOCK_HEIGHT = 20

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        self.grid_offset_x = self.SCREEN_WIDTH // 2
        self.grid_offset_y = self.SCREEN_HEIGHT // 2 - 80

        self.level_layout = [
            "WWWWWWWWWW",
            "W........W",
            "W.T.C.T..W",
            "W........W",
            "W..C.P.C.W",
            "W........W",
            "W.T.C.T..W",
            "W........W",
            "W........W",
            "WWWWWWWWWW",
        ]
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def _grid_to_screen(self, x, y):
        """Converts grid coordinates to screen pixel coordinates."""
        screen_x = self.grid_offset_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.grid_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, points, color):
        """Draws an anti-aliased polygon."""
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_block(self, surface, pos, color, top_color, height):
        """Draws a 3D isometric block."""
        x, y = pos
        px, py = self._grid_to_screen(x, y)

        # Points for the top face
        top_points = [
            (px, py - height),
            (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF - height),
            (px, py + self.TILE_HEIGHT_HALF * 2 - height),
            (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF - height),
        ]
        
        # Points for the side faces
        right_face_points = [
            (px, py + self.TILE_HEIGHT_HALF * 2 - height),
            (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF - height),
            (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF),
            (px, py + self.TILE_HEIGHT_HALF * 2)
        ]
        left_face_points = [
            (px, py + self.TILE_HEIGHT_HALF * 2 - height),
            (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF - height),
            (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF),
            (px, py + self.TILE_HEIGHT_HALF * 2)
        ]
        
        side_color = tuple(max(0, c - 30) for c in color)
        
        self._draw_iso_poly(surface, right_face_points, side_color)
        self._draw_iso_poly(surface, left_face_points, side_color)
        self._draw_iso_poly(surface, top_points, top_color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.move_cooldown = 0

        self.wall_positions = set()
        self.target_positions = []
        self.crate_positions = []
        
        for y, row in enumerate(self.level_layout):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.wall_positions.add(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char == 'C':
                    self.crate_positions.append(pos)
                elif char == 'T':
                    self.target_positions.append(pos)
        
        # Make crate positions a list of lists for mutability
        self.crate_positions = [list(pos) for pos in self.crate_positions]
        self.player_pos = list(self.player_pos)
        
        self.crates_on_target_at_reset = self._count_crates_on_targets()

        return self._get_observation(), self._get_info()

    def _is_stuck(self, crate_idx):
        """Check if a crate is in an unrecoverable corner."""
        pos = tuple(self.crate_positions[crate_idx])
        x, y = pos
        
        is_wall = lambda p: p in self.wall_positions
        
        # Check for simple corners
        up, down, left, right = (x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)
        
        if (is_wall(up) and is_wall(left)) or \
           (is_wall(up) and is_wall(right)) or \
           (is_wall(down) and is_wall(left)) or \
           (is_wall(down) and is_wall(right)):
            return True
            
        return False

    def step(self, action):
        reward = -0.01  # Small penalty for every step to encourage speed
        
        self.time_remaining -= 1.0 / self.FPS
        self.steps += 1
        
        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        
        if movement != 0 and self.move_cooldown == 0:
            self.move_cooldown = 5 # 5 frames cooldown
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            player_x, player_y = self.player_pos
            next_player_pos = (player_x + dx, player_y + dy)

            # --- Collision and Movement Logic ---
            if next_player_pos in self.wall_positions:
                # Player bumps into a wall
                # SFX: bump_wall.wav
                pass
            elif list(next_player_pos) in self.crate_positions:
                crate_idx = self.crate_positions.index(list(next_player_pos))
                next_crate_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)
                
                if next_crate_pos in self.wall_positions or list(next_crate_pos) in self.crate_positions:
                    # Crate is blocked
                    # SFX: bump_crate.wav
                    pass
                else:
                    # Successful push
                    # SFX: push_crate.wav
                    self.player_pos = list(next_player_pos)
                    
                    was_on_target = tuple(self.crate_positions[crate_idx]) in self.target_positions
                    self.crate_positions[crate_idx] = list(next_crate_pos)
                    is_on_target = tuple(self.crate_positions[crate_idx]) in self.target_positions
                    
                    if not was_on_target and is_on_target:
                        reward += 1.0 # Crate moved onto a target
                        # SFX: target_success.wav
                    elif was_on_target and not is_on_target:
                        reward -= 1.0 # Crate moved off a target
                    
                    if self._is_stuck(crate_idx) and not is_on_target:
                        reward -= 2.0 # Penalty for getting a crate stuck
            else:
                # Move to empty space
                self.player_pos = list(next_player_pos)

        # --- Update Game State ---
        self.score += reward
        
        # --- Termination Check ---
        crates_on_target = self._count_crates_on_targets()
        
        if crates_on_target == len(self.target_positions):
            self.win = True
            self.game_over = True
            reward += 50.0 # Big reward for winning
            # SFX: win_jingle.wav
        
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            if not self.win:
                self.game_over = True
                reward -= 50.0 # Big penalty for losing
                # SFX: lose_buzzer.wav
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _count_crates_on_targets(self):
        return sum(1 for c in self.crate_positions if tuple(c) in self.target_positions)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "crates_on_target": self._count_crates_on_targets(),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Floor and Targets ---
        for y in range(self.GRID_SIZE_Y):
            for x in range(self.GRID_SIZE_X):
                if (x, y) not in self.wall_positions:
                    px, py = self._grid_to_screen(x, y)
                    floor_points = [
                        (px, py),
                        (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF),
                        (px, py + self.TILE_HEIGHT_HALF * 2),
                        (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF)
                    ]
                    self._draw_iso_poly(self.screen, floor_points, self.COLOR_FLOOR)
                    
                    if (x, y) in self.target_positions:
                        is_active = list((x, y)) in self.crate_positions
                        color = self.COLOR_TARGET_ACTIVE if is_active else self.COLOR_TARGET
                        pygame.gfxdraw.filled_circle(self.screen, px, py + self.TILE_HEIGHT_HALF, 8, color)
                        pygame.gfxdraw.aacircle(self.screen, px, py + self.TILE_HEIGHT_HALF, 8, color)

        # --- Prepare and Sort Entities for Drawing ---
        entities = []
        for pos in self.wall_positions:
            entities.append({'type': 'wall', 'pos': pos})
        for i, pos in enumerate(self.crate_positions):
            entities.append({'type': 'crate', 'pos': pos, 'id': i})
        entities.append({'type': 'player', 'pos': self.player_pos})
        
        # Sort by grid y-coordinate, then x, for correct isometric rendering
        entities.sort(key=lambda e: (e['pos'][1], e['pos'][0]))

        # --- Draw Entities ---
        for entity in entities:
            pos = entity['pos']
            px, py = self._grid_to_screen(pos[0], pos[1])
            
            # Shadow
            shadow_points = [
                (px, py),
                (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF),
                (px, py + self.TILE_HEIGHT_HALF * 2),
                (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF)
            ]
            self._draw_iso_poly(self.screen, shadow_points, self.COLOR_SHADOW)

            # Entity
            if entity['type'] == 'wall':
                self._draw_iso_block(self.screen, pos, self.COLOR_WALL, self.COLOR_WALL_TOP, self.BLOCK_HEIGHT)
            elif entity['type'] == 'crate':
                color = self.COLOR_CRATE
                top_color = tuple(min(255, c + 30) for c in color)
                self._draw_iso_block(self.screen, pos, color, top_color, self.BLOCK_HEIGHT)
            elif entity['type'] == 'player':
                color = self.COLOR_PLAYER
                top_color = tuple(min(255, c + 30) for c in color)
                self._draw_iso_block(self.screen, pos, color, top_color, self.BLOCK_HEIGHT)

    def _render_ui(self):
        # Crates on Target
        crates_text = f"Crates: {self._count_crates_on_targets()} / {len(self.target_positions)}"
        text_surf = self.font_ui.render(crates_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (15, 15))

        # Timer
        time_left = max(0, int(self.time_remaining))
        timer_text = f"Time: {time_left}"
        timer_color = self.COLOR_TEXT
        if time_left < 10: timer_color = self.COLOR_TIMER_CRIT
        elif time_left < 20: timer_color = self.COLOR_TIMER_WARN
        text_surf = self.font_ui.render(timer_text, True, timer_color)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 15, 15))
        
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "TIME'S UP!"
            color = self.COLOR_TARGET_ACTIVE if self.win else self.COLOR_TIMER_CRIT
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a persistent key state dictionary
    key_state = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 0,
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 0,
        pygame.K_SPACE: 0,
        pygame.K_LSHIFT: 0,
        pygame.K_RSHIFT: 0,
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Sokoban")
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    running = True
    while running:
        # --- Human Input Processing ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_state:
                    key_state[event.key] = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
            elif event.type == pygame.KEYUP:
                if event.key in key_state:
                    key_state[event.key] = 0

        # --- Action Mapping ---
        movement = 0 # No-op
        if key_state[pygame.K_UP]: movement = 1
        elif key_state[pygame.K_DOWN]: movement = 2
        elif key_state[pygame.K_LEFT]: movement = 3
        elif key_state[pygame.K_RIGHT]: movement = 4
        
        space_held = key_state[pygame.K_SPACE]
        shift_held = key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT]
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            done = True
        
        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it.
        # We need to transpose it back for pygame's `surfarray.blit_array`
        render_obs = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(render_screen, render_obs)
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()