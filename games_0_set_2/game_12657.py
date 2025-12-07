import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:08:01.545943
# Source Brief: brief_02657.md
# Brief Index: 2657
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import time

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a growing maze where you can only turn left. Reach the exit before time runs out, without crossing your own path."
    )
    user_guide = (
        "Controls: Press SPACE to move forward and SHIFT to turn left. Reach the exit without trapping yourself."
    )
    auto_advance = True

    # --- Constants ---
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (220, 220, 240)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_EXIT = (255, 80, 80)
    COLOR_EXIT_GLOW = (255, 80, 80, 70)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PATH = (40, 40, 60)

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    TIME_LIMIT_SECONDS = 45
    MAX_EPISODE_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.player_pos = None
        self.player_dir = None
        self.exit_pos = None
        self.walls = None
        self.path_taken = None

        self.steps = 0
        self.score = 0
        self.start_ticks = 0
        self.time_remaining = 0.0
        self.game_over = False
        self.last_reward = 0.0

        # --- Camera & Visuals ---
        self.camera_offset = pygame.Vector2(0, 0)
        self.camera_zoom = 1.0
        self.target_camera_offset = pygame.Vector2(0, 0)
        self.target_camera_zoom = 1.0

        # A call to reset() is needed to initialize the game state for the first time.
        # However, we don't want to return values from __init__.
        # So we don't call self.reset() here, but it will be called by the user.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward = 0.0

        # Player state
        self.player_pos = pygame.Vector2(0, 0)
        self.player_dir = 1  # 0:Up, 1:Right, 2:Down, 3:Left

        # Maze state
        exit_dist = 15
        angle = self.np_random.uniform(0, 2 * math.pi)
        self.exit_pos = pygame.Vector2(
            round(exit_dist * math.cos(angle)),
            round(exit_dist * math.sin(angle))
        )
        
        self.walls = set()
        self.path_taken = {tuple(self.player_pos)}

        # Timer
        self.start_ticks = pygame.time.get_ticks()
        self.time_remaining = self.TIME_LIMIT_SECONDS

        # Reset camera to starting position
        self.camera_zoom = 2.0
        self.target_camera_zoom = 2.0
        self.camera_offset, self.target_camera_zoom = self._get_target_camera_transform()
        self.target_camera_offset = self.camera_offset

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        # --- Handle Actions ---
        # Action mapping: shift=turn left, space=move forward. Turn has priority.
        turn_action = action[2] == 1
        move_action = action[1] == 1

        action_taken = False
        if turn_action:
            # Sfx: Turn click
            self.player_dir = (self.player_dir - 1 + 4) % 4
            action_taken = True
            reward -= 0.01 # Small cost for turning

        elif move_action:
            action_taken = True
            dist_before = self.player_pos.distance_to(self.exit_pos)
            
            # Calculate next position
            move_vec = self._get_move_vector()
            next_pos = self.player_pos + move_vec
            
            # Create wall segment behind player
            wall_start = self.player_pos
            wall_end = next_pos
            
            # Ensure consistent wall representation
            wall = tuple(sorted((tuple(wall_start), tuple(wall_end))))

            # Check for collision
            if wall in self.walls or tuple(next_pos) in self.path_taken:
                # Sfx: Collision/Stuck sound
                self.game_over = True
                reward = -100
            else:
                # Sfx: Move forward woosh
                self.walls.add(wall)
                self.player_pos = next_pos
                self.path_taken.add(tuple(self.player_pos))
                
                dist_after = self.player_pos.distance_to(self.exit_pos)
                reward += (dist_before - dist_after) # Reward for getting closer

        if not action_taken:
            reward -= 0.05 # Penalty for inaction

        # --- Update Game State ---
        self._update_timer()
        
        # --- Check Termination Conditions ---
        terminated = self.game_over or self._check_termination_conditions()
        
        # --- Calculate Final Reward ---
        if terminated and not self.game_over:
            if self.player_pos.distance_to(self.exit_pos) < 0.1:
                # Sfx: Win fanfare
                reward += 100
            elif self.time_remaining <= 0:
                # Sfx: Time out buzzer
                reward = -100
            elif self.steps >= self.MAX_EPISODE_STEPS:
                reward = -50
        
        self.game_over = terminated
        self.score += reward
        self.last_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_move_vector(self):
        if self.player_dir == 0: return pygame.Vector2(0, -1)  # Up
        if self.player_dir == 1: return pygame.Vector2(1, 0)   # Right
        if self.player_dir == 2: return pygame.Vector2(0, 1)   # Down
        if self.player_dir == 3: return pygame.Vector2(-1, 0)  # Left
        return pygame.Vector2(0, 0)

    def _update_timer(self):
        elapsed_seconds = (pygame.time.get_ticks() - self.start_ticks) / 1000
        self.time_remaining = max(0, self.TIME_LIMIT_SECONDS - elapsed_seconds)

    def _check_termination_conditions(self):
        if self.player_pos.distance_to(self.exit_pos) < 0.1:
            return True
        if self.time_remaining <= 0:
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "distance_to_exit": self.player_pos.distance_to(self.exit_pos),
        }

    def _world_to_screen(self, pos):
        """Converts grid coordinates to screen coordinates using camera."""
        screen_pos = (pos * self.GRID_SIZE) * self.camera_zoom + self.camera_offset
        return int(screen_pos.x), int(screen_pos.y)

    def _get_target_camera_transform(self):
        """Calculate the ideal camera zoom and offset to frame the maze."""
        points = [self.player_pos, self.exit_pos]
        if self.walls:
            for w in self.walls:
                points.append(pygame.Vector2(w[0]))
                points.append(pygame.Vector2(w[1]))
        
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        world_w = (max_x - min_x + 2) * self.GRID_SIZE
        world_h = (max_y - min_y + 2) * self.GRID_SIZE
        
        if world_w == 0 or world_h == 0:
             return self.camera_offset, self.camera_zoom

        zoom_x = self.SCREEN_WIDTH / world_w if world_w > 0 else 1
        zoom_y = self.SCREEN_HEIGHT / world_h if world_h > 0 else 1
        target_zoom = min(zoom_x, zoom_y, 2.5) # Cap max zoom

        center_x = (min_x + max_x) / 2 * self.GRID_SIZE
        center_y = (min_y + max_y) / 2 * self.GRID_SIZE

        target_offset_x = self.SCREEN_WIDTH / 2 - center_x * target_zoom
        target_offset_y = self.SCREEN_HEIGHT / 2 - center_y * target_zoom
        
        return pygame.Vector2(target_offset_x, target_offset_y), target_zoom

    def _update_camera(self):
        """Smoothly interpolate camera to target position."""
        self.target_camera_offset, self.target_camera_zoom = self._get_target_camera_transform()
        
        lerp_factor = 0.08
        self.camera_offset.x += (self.target_camera_offset.x - self.camera_offset.x) * lerp_factor
        self.camera_offset.y += (self.target_camera_offset.y - self.camera_offset.y) * lerp_factor
        self.camera_zoom += (self.target_camera_zoom - self.camera_zoom) * lerp_factor

    def _get_observation(self):
        self._update_camera()
        
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render path taken
        for pos in self.path_taken:
            screen_pos = self._world_to_screen(pygame.Vector2(pos))
            size = max(2, int(self.GRID_SIZE * self.camera_zoom * 0.2))
            pygame.draw.rect(self.screen, self.COLOR_PATH, (screen_pos[0] - size//2, screen_pos[1] - size//2, size, size))
            
        # Render walls
        wall_thickness = max(1, int(2 * self.camera_zoom))
        for wall in self.walls:
            start_pos = self._world_to_screen(pygame.Vector2(wall[0]))
            end_pos = self._world_to_screen(pygame.Vector2(wall[1]))
            pygame.draw.line(self.screen, self.COLOR_WALL, start_pos, end_pos, wall_thickness)

        # Render exit
        exit_size = int(self.GRID_SIZE * self.camera_zoom * 0.8)
        exit_screen_pos = self._world_to_screen(self.exit_pos)
        exit_rect = pygame.Rect(exit_screen_pos[0] - exit_size // 2, exit_screen_pos[1] - exit_size // 2, exit_size, exit_size)
        self._draw_glow_rect(self.screen, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, exit_rect, 15)

        # Render player
        player_size = int(self.GRID_SIZE * self.camera_zoom * 0.7)
        player_screen_pos = self._world_to_screen(self.player_pos)
        player_rect = pygame.Rect(player_screen_pos[0] - player_size // 2, player_screen_pos[1] - player_size // 2, player_size, player_size)
        self._draw_glow_rect(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, player_rect, 20)

        # Render player direction indicator
        move_vec = self._get_move_vector()
        indicator_start = pygame.Vector2(player_screen_pos)
        indicator_end = indicator_start + move_vec * player_size * 0.7
        pygame.draw.line(self.screen, self.COLOR_PLAYER, indicator_start, indicator_end, max(1, int(3 * self.camera_zoom)))

    def _draw_glow_rect(self, surface, color, glow_color, rect, radius):
        """Draws a rectangle with a soft glow effect."""
        for i in range(radius, 0, -1):
            alpha = int(glow_color[3] * (1 - i / radius))
            if alpha > 0:
                temp_color = (glow_color[0], glow_color[1], glow_color[2], alpha)
                glow_rect = rect.inflate(i*2, i*2)
                shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(shape_surf, temp_color, (0, 0, *glow_rect.size), border_radius=int(i/2))
                surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=2)

    def _render_ui(self):
        # Timer display
        time_str = f"TIME: {self.time_remaining:.1f}"
        time_surf = self.font_large.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score display
        score_str = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_str, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 40))

        # Last reward display
        reward_color = self.COLOR_PLAYER if self.last_reward > 0 else self.COLOR_EXIT if self.last_reward < 0 else self.COLOR_TEXT
        reward_str = f"REWARD: {self.last_reward:+.2f}"
        reward_surf = self.font_small.render(reward_str, True, reward_color)
        self.screen.blit(reward_surf, (self.SCREEN_WIDTH - reward_surf.get_width() - 10, 10))
        
        # Steps display
        steps_str = f"STEPS: {self.steps}"
        steps_surf = self.font_small.render(steps_str, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 10, 35))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for developer convenience and is not part of the Gym API
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in the hosted environment but is useful for local testing.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    env.validate_implementation() # Run validation
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display for rendering
    pygame.display.set_caption("Left Turn Maze")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_score = 0
    
    # Action mapping for human player
    # SHIFT -> Turn Left, SPACE -> Move Forward
    
    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("Q: Quit")
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_score = 0
                    terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            # Note: This simple input handling doesn't queue actions,
            # so holding both keys might only register one.
            # The environment gives priority to turning.
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1 # Shift pressed
            elif keys[pygame.K_SPACE]:
                action[1] = 1 # Space pressed
                
            obs, reward, terminated, truncated, info = env.step(action)
            total_score += reward
        
        # Blit the observation from the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate for playability
        env.clock.tick(10) # Slower for turn-based feel
        
    env.close()