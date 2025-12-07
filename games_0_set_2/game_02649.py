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
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character and push crates."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced Sokoban variant. Push all the crates onto the green targets "
        "before the 60-second timer runs out. Complete all 3 levels to win."
    )

    # Frames auto-advance for real-time timer and smooth animations.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GRID_SIZE = 40
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_WALL = (68, 71, 90)
    COLOR_TARGET = (80, 250, 123, 100) # Green, with alpha
    COLOR_CRATE = (255, 184, 108)
    COLOR_CRATE_ON_TARGET = (139, 233, 253) # Cyan
    COLOR_PLAYER = (255, 85, 85)
    COLOR_PLAYER_OUTLINE = (255, 121, 121)
    COLOR_TEXT = (248, 248, 242)
    COLOR_TIMER_WARN = (255, 121, 121)

    LEVEL_TIME = 60  # seconds per level
    MAX_STEPS = LEVEL_TIME * FPS * 3 # 3 levels
    MOVE_ANIMATION_FRAMES = 5 # How many frames a move takes

    # Level maps
    LEVELS = [
        [
            "WWWWWWWWWWWWWWWW",
            "W              W",
            "W  P  C     T  W",
            "W              W",
            "W    WWWWWW    W",
            "W T      C   T W",
            "W    WWWWWW    W",
            "W              W",
            "W  C     C   T W",
            "WWWWWWWWWWWWWWWW",
        ],
        [
            "WWWWWWWWWWWWWWWW",
            "W T  W   W C   W",
            "W C  W P W T   W",
            "W T  WWWWW C   W",
            "W C      T   C W",
            "W T      W   T W",
            "WWWWWWWWWWWWWWWW",
            "W              W",
            "W              W",
            "WWWWWWWWWWWWWWWW",
        ],
        [
            "WWWWWWWWWWWWWWWW",
            "WPC W W      T W",
            "W C W W  C C T W",
            "W   W WWWWWW W W",
            "W C W      W   W",
            "W T WWWWWW W C W",
            "W   W    W W T W",
            "W T W    W   T W",
            "W   W    WWWWWWW",
            "WWWWWWWWWWWWWWWW",
        ]
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Game state variables
        self.player_pos = np.array([0, 0], dtype=float)
        self.player_visual_pos = np.array([0, 0], dtype=float)
        self.walls = []
        self.targets = []
        self.crates = [] # List of dicts: {'pos': np.array, 'visual_pos': np.array, 'id': int}
        
        self.current_level = 0
        self.level_time_remaining = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.is_moving = False
        self.animation_frame = 0
        self.move_origin = None
        self.move_target = None
        self.pushed_crate = None

        # self.reset() is called by the test harness, no need to call it here.
        
    def _load_level(self, level_index):
        self.walls.clear()
        self.targets.clear()
        self.crates.clear()
        
        level_map = self.LEVELS[level_index]
        self.map_height = len(level_map)
        self.map_width = len(level_map[0])

        offset_x = (self.SCREEN_WIDTH - self.map_width * self.GRID_SIZE) // 2
        offset_y = (self.SCREEN_HEIGHT - self.map_height * self.GRID_SIZE) // 2

        crate_id = 0
        for r, row in enumerate(level_map):
            for c, char in enumerate(row):
                grid_pos = np.array([c, r])
                pixel_pos = grid_pos * self.GRID_SIZE + np.array([offset_x, offset_y])

                if char == 'W':
                    self.walls.append(grid_pos)
                elif char == 'P':
                    self.player_pos = grid_pos.astype(float)
                    self.player_visual_pos = self.player_pos * self.GRID_SIZE + np.array([offset_x, offset_y])
                elif char == 'C':
                    self.crates.append({
                        'pos': grid_pos.astype(float),
                        'visual_pos': pixel_pos.astype(float),
                        'id': crate_id
                    })
                    crate_id += 1
                elif char == 'T':
                    self.targets.append(grid_pos)

        # In case a level has both C and T on the same spot
        target_crate_positions = [c['pos'] for c in self.crates]
        for r, row in enumerate(level_map):
            for c, char in enumerate(row):
                grid_pos = np.array([c, r])
                if char == 'T' and not any(np.array_equal(grid_pos, pos) for pos in target_crate_positions):
                     # Re-add targets that weren't under crates initially
                     if not any(np.array_equal(grid_pos, t) for t in self.targets):
                         self.targets.append(grid_pos)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 0
        
        self._load_level(self.current_level)
        self.level_time_remaining = self.LEVEL_TIME

        self.is_moving = False
        self.animation_frame = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0
        
        # --- Update Timer ---
        self.level_time_remaining -= 1 / self.FPS
        if self.level_time_remaining <= 0:
            self.game_over = True
            reward -= 100 # Timeout penalty
            return self._get_observation(), reward, True, False, self._get_info()

        # --- Handle Animation ---
        if self.is_moving:
            self.animation_frame += 1
            progress = self.animation_frame / self.MOVE_ANIMATION_FRAMES
            
            offset_x = (self.SCREEN_WIDTH - self.map_width * self.GRID_SIZE) // 2
            offset_y = (self.SCREEN_HEIGHT - self.map_height * self.GRID_SIZE) // 2
            grid_offset = np.array([offset_x, offset_y])

            # Interpolate player
            start_pixel = self.move_origin * self.GRID_SIZE + grid_offset
            end_pixel = self.move_target * self.GRID_SIZE + grid_offset
            self.player_visual_pos = start_pixel + (end_pixel - start_pixel) * progress

            # Interpolate pushed crate
            if self.pushed_crate is not None:
                crate_start_pixel = self.pushed_crate['start_pos'] * self.GRID_SIZE + grid_offset
                crate_end_pixel = self.pushed_crate['end_pos'] * self.GRID_SIZE + grid_offset
                self.pushed_crate['crate']['visual_pos'] = crate_start_pixel + (crate_end_pixel - crate_start_pixel) * progress

            if self.animation_frame >= self.MOVE_ANIMATION_FRAMES:
                self.is_moving = False
                self.pushed_crate = None
        
        # --- Handle Actions ---
        if not self.is_moving:
            movement = action[0]
            
            if movement != 0: # If there is a move action
                move_dir = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}.get(movement)
                if move_dir:
                    move_dir = np.array(move_dir)
                    target_pos = self.player_pos + move_dir
                    
                    is_wall_collision = any(np.array_equal(target_pos, w) for w in self.walls)
                    
                    crate_to_push = None
                    for c in self.crates:
                        if np.array_equal(target_pos, c['pos']):
                            crate_to_push = c
                            break

                    can_move = False
                    if not is_wall_collision:
                        if crate_to_push:
                            # Check space behind crate
                            behind_crate_pos = crate_to_push['pos'] + move_dir
                            is_blocked = any(np.array_equal(behind_crate_pos, w) for w in self.walls) or \
                                         any(np.array_equal(behind_crate_pos, other_c['pos']) for other_c in self.crates)
                            if not is_blocked:
                                can_move = True
                                
                                # --- Calculate Reward for Crate Push ---
                                crates_on_target_before = self._count_crates_on_target()
                                dist_before = self._get_crate_dist_to_nearest_target(crate_to_push)

                                # Update crate logical position
                                self.pushed_crate = {
                                    'crate': crate_to_push,
                                    'start_pos': crate_to_push['pos'].copy(),
                                    'end_pos': behind_crate_pos.copy()
                                }
                                crate_to_push['pos'] = behind_crate_pos

                                dist_after = self._get_crate_dist_to_nearest_target(crate_to_push)
                                crates_on_target_after = self._count_crates_on_target()

                                if dist_after < dist_before:
                                    reward += 0.1 # Moved closer to a target
                                elif dist_after > dist_before:
                                    reward -= 0.1 # Moved further away

                                if crates_on_target_after > crates_on_target_before:
                                    reward += 1.0 # Placed a crate on a target
                                elif crates_on_target_after < crates_on_target_before:
                                    reward -= 1.0 # Moved a crate off a target
                        else:
                            can_move = True

                    if can_move:
                        # Start animation
                        self.is_moving = True
                        self.animation_frame = 0
                        self.move_origin = self.player_pos.copy()
                        self.move_target = target_pos.copy()
                        self.player_pos = target_pos # Update logical position immediately
                        
        # --- Check for Level Completion ---
        if self._count_crates_on_target() == len(self.crates):
            self.score += 100 # Level complete bonus
            reward += 100
            self.current_level += 1
            if self.current_level >= len(self.LEVELS):
                self.game_over = True # All levels complete
            else:
                self._load_level(self.current_level)
                self.level_time_remaining = self.LEVEL_TIME
                self.is_moving = False # Stop any ongoing animations

        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _count_crates_on_target(self):
        count = 0
        crate_positions = [tuple(c['pos']) for c in self.crates]
        target_positions = [tuple(t) for t in self.targets]
        for cp in crate_positions:
            if cp in target_positions:
                count += 1
        return count

    def _get_crate_dist_to_nearest_target(self, crate):
        crate_pos = crate['pos']
        if not self.targets: return 0
        
        # Find distance to closest *unoccupied* target
        occupied_targets = {tuple(c['pos']) for c in self.crates if c['id'] != crate['id'] and tuple(c['pos']) in [tuple(t) for t in self.targets]}
        unoccupied_targets = [t for t in self.targets if tuple(t) not in occupied_targets]

        if not unoccupied_targets: # All targets are occupied by other crates
            # Default to first target if all occupied, or any target if none are
            return np.linalg.norm(crate_pos - self.targets[0])

        distances = [np.linalg.norm(crate_pos - t) for t in unoccupied_targets]
        return min(distances)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        offset_x = (self.SCREEN_WIDTH - self.map_width * self.GRID_SIZE) // 2
        offset_y = (self.SCREEN_HEIGHT - self.map_height * self.GRID_SIZE) // 2
        
        # Render targets
        for t_pos in self.targets:
            pixel_pos = t_pos * self.GRID_SIZE + np.array([offset_x, offset_y])
            target_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            target_surface.fill(self.COLOR_TARGET)
            self.screen.blit(target_surface, (int(pixel_pos[0]), int(pixel_pos[1])))

        # Render walls
        for wall_pos in self.walls:
            pixel_pos = wall_pos * self.GRID_SIZE + np.array([offset_x, offset_y])
            wall_rect = pygame.Rect(int(pixel_pos[0]), int(pixel_pos[1]), self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall_rect)

        # Render crates
        target_tuples = {tuple(t) for t in self.targets}
        for crate in self.crates:
            is_on_target = tuple(crate['pos']) in target_tuples
            color = self.COLOR_CRATE_ON_TARGET if is_on_target else self.COLOR_CRATE
            pos = crate['visual_pos']
            rect = pygame.Rect(int(pos[0]) + 4, int(pos[1]) + 4, self.GRID_SIZE - 8, self.GRID_SIZE - 8)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Render player
        player_center_x = int(self.player_visual_pos[0] + self.GRID_SIZE / 2)
        player_center_y = int(self.player_visual_pos[1] + self.GRID_SIZE / 2)
        radius = self.GRID_SIZE // 3
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, radius + 2, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Level text
        level_text = self.font_large.render(f"Level: {self.current_level + 1}/{len(self.LEVELS)}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer text
        time_left = max(0, self.level_time_remaining)
        time_color = self.COLOR_TIMER_WARN if time_left < 10 and int(time_left * 2) % 2 == 0 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"Time: {time_left:.1f}", True, time_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Crates on target text
        crates_on_target = self._count_crates_on_target()
        total_crates = len(self.crates)
        crates_text = self.font_small.render(f"Crates: {crates_on_target}/{total_crates}", True, self.COLOR_TEXT)
        self.screen.blit(crates_text, (self.SCREEN_WIDTH - crates_text.get_width() - 10, 50))
        
        # Score text
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 50))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level + 1,
            "time_remaining": self.level_time_remaining,
            "crates_on_target": self._count_crates_on_target(),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run by the verifier
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    # Create a window to display the game
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Rush")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Optional: wait for a key press to reset
            # running = False 
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            
    env.close()